"""Two-GPU smoke test for SGLang's checkpointable native collectives.

Run with:
  python -m torch.distributed.run --standalone --nproc-per-node=2 \
    test/manual/distributed/test_checkpoint_collectives.py
"""

import os
from pathlib import Path

import torch
import torch.distributed as dist


def assert_reduce(actual: torch.Tensor, rank: int, iteration: int) -> None:
    expected = torch.full_like(actual, 2 * iteration + 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if rank == 0:
        print(f"iteration={iteration} value={actual[0].item()}", flush=True)


def assert_gather(actual: torch.Tensor, iteration: int) -> None:
    expected = torch.tensor(
        [iteration + 1, iteration + 2], dtype=actual.dtype, device=actual.device
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def main() -> None:
    os.environ["SGLANG_CRIU_SUSPEND_DEVICE_PROCESS_GROUP"] = "1"
    os.environ["SGLANG_CRIU_ALL_REDUCE_MAX_BYTES"] = str(1 << 20)
    os.environ["SGLANG_CRIU_ALL_GATHER_MAX_ELEMS"] = "1024"

    from sglang.srt.distributed.device_communicators import custom_all_reduce
    from sglang.srt.distributed.device_communicators.checkpoint_collectives import (
        NativeCheckpointCollectives,
        SymmetricAllGather,
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    cpu_group = dist.new_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # The engine initializes SGLang's world coordinator before constructing
    # communicators. This focused test starts from raw torch.distributed, so
    # declare the already-selected same-node B200 pair as P2P-capable.
    custom_all_reduce.can_use_custom_all_reduce_with_nvlink = lambda **_: True
    all_reduce = custom_all_reduce.CustomAllreduce(cpu_group, device, max_size=1 << 20)
    all_gather = SymmetricAllGather(cpu_group, device, max_elems=1024)
    collectives = NativeCheckpointCollectives(all_reduce, all_gather)

    graph_input = torch.empty(2560, dtype=torch.bfloat16, device=device)
    gather_input = torch.empty(1, dtype=torch.bfloat16, device=device)
    graph = torch.cuda.CUDAGraph()
    dist.barrier(group=cpu_group)
    with torch.cuda.graph(graph), all_reduce.capture():
        graph_output = all_reduce.custom_all_reduce(graph_input)
        gather_output = collectives.all_gather(gather_input)

    def replay(iteration: int) -> None:
        graph_input.fill_(local_rank + iteration + 1)
        gather_input.fill_(local_rank + iteration + 1)
        torch.cuda.synchronize(device)
        dist.barrier(group=cpu_group)
        graph.replay()
        torch.cuda.synchronize(device)
        assert_reduce(graph_output, local_rank, iteration)
        assert_gather(gather_output, iteration)

    replay(1)
    replay(2)

    store_path = Path(f"/tmp/sglang-native-renew-{os.environ['MASTER_PORT']}")
    if rank == 0:
        store_path.unlink(missing_ok=True)
    dist.barrier(group=cpu_group)
    collectives.prepare_checkpoint()
    collectives.set_control_group(None)
    dist.destroy_process_group(cpu_group)
    dist.destroy_process_group()

    store = dist.FileStore(str(store_path), world_size)
    dist.init_process_group("nccl", store=store, rank=rank, world_size=world_size)
    cpu_group = dist.new_group(backend="gloo")
    collectives.set_control_group(cpu_group)
    collectives.restore_after_checkpoint()

    replay(3)
    replay(4)

    collectives.close()
    all_reduce.close()
    dist.destroy_process_group(cpu_group)
    dist.destroy_process_group()
    if rank == 0:
        store_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
