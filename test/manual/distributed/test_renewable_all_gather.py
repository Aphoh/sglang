"""Two-GPU smoke test for all-gather across CPU-group renewal.

Run with:
  PYTHONPATH=python torchrun --standalone --nproc-per-node=2 \
    test/manual/distributed/test_renewable_all_gather.py
"""

import os
from datetime import timedelta

import torch
import torch.distributed as dist

from sglang.srt.distributed.cpu_group_lifecycle import (
    CpuGroupBinding,
    CpuGroupRecipe,
    CpuGroupState,
    CpuGroupTransaction,
)
from sglang.srt.distributed.device_communicators.renewable_all_gather import (
    RenewableAllGather,
)


def expected(input_: torch.Tensor, value: int, world_size: int) -> torch.Tensor:
    return torch.cat(
        [
            torch.full_like(input_, value + source_rank)
            for source_rank in range(world_size)
        ]
    )


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    ranks = tuple(range(world_size))
    group = dist.new_group(ranks=list(ranks), backend="gloo")
    binding = CpuGroupBinding(
        name="renewable_all_gather_test",
        recipe=CpuGroupRecipe(
            group_ranks=(ranks,),
            torch_distributed_backend="gloo",
            gloo_timeout=timedelta(seconds=120),
            model_parallel_timeout=None,
        ),
        group=group,
        active_ranks=torch.ones(world_size, dtype=torch.int32),
    )
    all_gather = RenewableAllGather(binding, device, max_bytes=1 << 20)
    pointers = all_gather.memory.ptrs

    eager_input = torch.full((2560,), rank + 1, dtype=torch.bfloat16, device=device)
    torch.testing.assert_close(
        all_gather.all_gather(eager_input),
        expected(eager_input, 1, world_size),
        rtol=0,
        atol=0,
    )

    graph_input = torch.empty(2560, dtype=torch.bfloat16, device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = all_gather.all_gather(graph_input)

    def replay(value: int) -> None:
        graph_input.fill_(rank + value)
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(
            graph_output,
            expected(graph_input, value, world_size),
            rtol=0,
            atol=0,
        )

    replay(3)
    transaction = CpuGroupTransaction([binding])
    transaction.suspend()
    assert binding.state is CpuGroupState.SUSPENDED
    assert not all_gather.memory.attached
    assert all_gather.memory.control_group is None

    transaction.resume()
    assert binding.state is CpuGroupState.ACTIVE
    assert all_gather.memory.attached
    assert all_gather.memory.ptrs == pointers
    replay(7)
    assert all_gather.status() == [0, 0, 0, 0]

    all_gather.close()
    binding.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
