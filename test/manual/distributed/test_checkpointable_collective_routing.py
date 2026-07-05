"""TP2 smoke test for GroupCoordinator checkpointable collective routing.

Run with:
  PYTHONPATH=python torchrun --standalone --nproc-per-node=2 \
    test/manual/distributed/test_checkpointable_collective_routing.py
"""

import os

import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as parallel_state
from sglang.srt.distributed.checkpoint_lifecycle import CheckpointLifecycle


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    world = parallel_state.init_world_group(
        list(range(world_size)), local_rank, "nccl", reuse_device_group=True
    )
    parallel_state._WORLD = world

    coordinator = parallel_state.init_model_parallel_group(
        [list(range(world_size))],
        local_rank,
        "nccl",
        group_name="checkpointable_collective_test",
        use_checkpointable_collectives=True,
    )
    assert coordinator.device_group is dist.group.WORLD
    assert not coordinator.owns_device_group
    assert coordinator.pynccl_comm is None
    assert coordinator.ca_comm is not None
    assert coordinator.ag_comm is not None

    all_reduce_input = torch.full(
        (2560,), rank + 1, dtype=torch.bfloat16, device=device
    )
    all_reduce_output = coordinator.all_reduce(all_reduce_input)
    torch.testing.assert_close(
        all_reduce_output,
        torch.full_like(all_reduce_output, sum(range(1, world_size + 1))),
        rtol=0,
        atol=0,
    )

    all_gather_input = torch.full(
        (2560,), rank + 1, dtype=torch.bfloat16, device=device
    )
    all_gather_output = torch.empty(
        world_size * all_gather_input.numel(),
        dtype=all_gather_input.dtype,
        device=device,
    )
    coordinator.all_gather_into_tensor(all_gather_output, all_gather_input)
    expected_gather = torch.cat(
        [torch.full_like(all_gather_input, peer + 1) for peer in range(world_size)]
    )
    torch.testing.assert_close(all_gather_output, expected_gather, rtol=0, atol=0)

    graph_reduce_input = torch.empty_like(all_reduce_input)
    graph_gather_input = torch.empty_like(all_gather_input)
    graph_gather_output = torch.empty_like(all_gather_output)
    graph = torch.cuda.CUDAGraph()
    with coordinator.ca_comm.capture(), torch.cuda.graph(graph):
        graph_reduce_output = coordinator.all_reduce(graph_reduce_input)
        coordinator.all_gather_into_tensor(graph_gather_output, graph_gather_input)

    def replay(value: int) -> None:
        graph_reduce_input.fill_(rank + value)
        graph_gather_input.fill_(rank + value)
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize(device)
        reduce_total = world_size * value + sum(range(world_size))
        torch.testing.assert_close(
            graph_reduce_output,
            torch.full_like(graph_reduce_output, reduce_total),
            rtol=0,
            atol=0,
        )
        expected = torch.cat(
            [
                torch.full_like(graph_gather_input, peer + value)
                for peer in range(world_size)
            ]
        )
        torch.testing.assert_close(graph_gather_output, expected, rtol=0, atol=0)

    replay(3)
    checkpoint = CheckpointLifecycle(
        [world, coordinator],
        store_prefix="/tmp/sglang-device-world-test",
        synchronize=lambda: torch.cuda.synchronize(device),
    )
    checkpoint.suspend()
    assert not dist.is_initialized()
    checkpoint.resume()
    replay(7)

    coordinator.destroy()
    world.destroy()
    parallel_state._WORLD = None
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
