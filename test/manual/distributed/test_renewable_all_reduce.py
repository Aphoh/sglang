"""Two-GPU smoke test for all-reduce across CPU-group renewal.

Run with:
  PYTHONPATH=python torchrun --standalone --nproc-per-node=2 \
    test/manual/distributed/test_renewable_all_reduce.py
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
from sglang.srt.distributed.device_communicators import custom_all_reduce


def expected(input_: torch.Tensor, value: int, world_size: int) -> torch.Tensor:
    total = world_size * value + sum(range(world_size))
    return torch.full_like(input_, total)


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
        name="renewable_all_reduce_test",
        recipe=CpuGroupRecipe(
            group_ranks=(ranks,),
            torch_distributed_backend="gloo",
            gloo_timeout=timedelta(seconds=120),
            model_parallel_timeout=None,
        ),
        group=group,
        active_ranks=torch.ones(world_size, dtype=torch.int32),
    )

    custom_all_reduce.can_use_custom_all_reduce_with_nvlink = lambda **_: True
    all_reduce = custom_all_reduce.CustomAllreduce(
        group,
        device,
        max_size=1 << 20,
        lifecycle=binding,
    )
    assert not all_reduce.disabled
    meta_pointers = tuple(all_reduce.meta_ptrs)
    buffer_pointers = tuple(all_reduce.buffer_ptrs)

    eager_input = torch.full((2560,), rank + 1, dtype=torch.bfloat16, device=device)
    eager_output = all_reduce.custom_all_reduce(eager_input)
    assert eager_output is not None
    torch.testing.assert_close(
        eager_output, expected(eager_input, 1, world_size), rtol=0, atol=0
    )

    graph_input = torch.empty(2560, dtype=torch.bfloat16, device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), all_reduce.capture():
        graph_output = all_reduce.custom_all_reduce(graph_input)
    assert graph_output is not None

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
    assert all_reduce.group is None
    assert not all_reduce.meta_memory.attached
    assert not all_reduce.buffer_memory.attached

    transaction.resume()
    assert binding.state is CpuGroupState.ACTIVE
    assert tuple(all_reduce.meta_ptrs) == meta_pointers
    assert tuple(all_reduce.buffer_ptrs) == buffer_pointers
    replay(7)

    all_reduce.close()
    binding.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
