"""Two-GPU smoke test for the checkpointable Movin NIXL communicator.

Run with:
  torchrun --standalone --nproc-per-node=2 \
    test/manual/distributed/test_movin_nixl_all_reduce.py
"""

import os

import torch
import torch.distributed as dist

from movin import TorchDistributedNixlAllReduce


def assert_result(actual: torch.Tensor, rank: int, iteration: int) -> None:
    expected_value = 2 * iteration + 3
    expected = torch.full_like(actual, expected_value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if rank == 0:
        print(f"iteration={iteration} value={actual[0].item()}", flush=True)


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    cpu_group = dist.new_group(backend="gloo")

    comm = TorchDistributedNixlAllReduce(
        cpu_group,
        device,
        "movin-test",
    )
    eager_input = torch.full(
        (2560,), local_rank + 1, dtype=torch.bfloat16, device=device
    )
    assert_result(comm.all_reduce(eager_input), local_rank, 0)

    graph_input = torch.empty_like(eager_input)
    graph = torch.cuda.CUDAGraph()
    dist.barrier(group=cpu_group)
    with torch.cuda.graph(graph), comm.capture():
        graph_output = comm.all_reduce(graph_input)

    for iteration in (1, 2):
        graph_input.fill_(local_rank + iteration + 1)
        torch.cuda.synchronize(device)
        dist.barrier(group=cpu_group)
        graph.replay()
        torch.cuda.synchronize(device)
        assert_result(graph_output, local_rank, iteration)
        dist.barrier(group=cpu_group)

    comm.prepare_criu()
    dist.barrier(group=cpu_group)
    comm.restore_after_criu()

    for iteration in (3, 4):
        graph_input.fill_(local_rank + iteration + 1)
        torch.cuda.synchronize(device)
        dist.barrier(group=cpu_group)
        graph.replay()
        torch.cuda.synchronize(device)
        assert_result(graph_output, local_rank, iteration)
        dist.barrier(group=cpu_group)

    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
