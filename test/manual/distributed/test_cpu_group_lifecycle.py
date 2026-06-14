"""Exercise repeated SGLang CPU process-group teardown and recreation.

Run with:
  torchrun --standalone --nproc-per-node=2 \
    test/manual/distributed/test_cpu_group_lifecycle.py
"""

import os

import torch
import torch.distributed as dist

from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    _register_group,
    resume_cpu_process_groups,
    resume_device_process_group,
    suspend_cpu_process_groups,
    suspend_device_process_group,
    wait_for_process_group_teardown,
)


class FakeMovinCollectives:
    def __init__(self, control_group):
        self.control_group = control_group

    def set_control_group(self, control_group) -> None:
        self.control_group = control_group


def make_cpu_only_coordinator(cpu_group) -> GroupCoordinator:
    coordinator = GroupCoordinator.__new__(GroupCoordinator)
    coordinator.unique_name = "cpu-lifecycle-test:0"
    coordinator.rank = dist.get_rank()
    coordinator.world_size = dist.get_world_size()
    coordinator.group_name = "cpu-lifecycle-test"
    coordinator.cpu_group = cpu_group
    coordinator.device_group = dist.group.WORLD
    coordinator._owns_device_group = False
    coordinator._device_group_is_cpu_alias = False
    coordinator._device_group_suspended = False
    coordinator._group_ranks = [list(range(dist.get_world_size()))]
    coordinator._torch_distributed_backend = "gloo"
    coordinator._gloo_timeout = torch.distributed.default_pg_timeout
    coordinator._recovered_rank = False
    coordinator._cpu_group_suspended = False
    coordinator._cpu_group_generation = 0
    coordinator.use_message_queue_broadcaster = False
    coordinator.mq_broadcaster = None

    for name in (
        "pynccl_comm",
        "pymscclpp_comm",
        "ca_comm",
        "qr_comm",
        "torch_symm_mem_comm",
        "hpu_communicator",
        "xpu_communicator",
        "npu_communicator",
    ):
        setattr(coordinator, name, None)
    coordinator.movin_collectives = FakeMovinCollectives(cpu_group)
    _register_group(coordinator)
    return coordinator


def assert_cpu_collective(coordinator: GroupCoordinator, cycle: int) -> None:
    value = torch.tensor([dist.get_rank() + cycle], dtype=torch.int64)
    dist.all_reduce(value, group=coordinator.cpu_group)
    expected = sum(range(dist.get_world_size())) + cycle * dist.get_world_size()
    assert value.item() == expected


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    os.environ["SGLANG_CRIU_SUSPEND_DEVICE_PROCESS_GROUP"] = "1"
    os.environ["SGLANG_CRIU_DEVICE_STORE"] = (
        f"/tmp/sglang-cpu-lifecycle-{os.environ['MASTER_PORT']}"
    )

    coordinator = make_cpu_only_coordinator(dist.new_group(backend="gloo"))
    graph_value = torch.zeros(1, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_value.add_(1)

    for cycle in range(4):
        assert_cpu_collective(coordinator, cycle)
        graph.replay()
        torch.cuda.synchronize()
        assert graph_value.item() == cycle + 1

        suspend_cpu_process_groups()
        assert coordinator.cpu_group is None
        assert coordinator.movin_collectives.control_group is None
        suspend_device_process_group()
        wait_for_process_group_teardown()
        assert not dist.is_initialized()
        assert coordinator.device_group is None

        resume_device_process_group()
        assert dist.is_initialized()
        assert coordinator.device_group is dist.group.WORLD
        resume_cpu_process_groups()
        assert (
            coordinator.cpu_group is coordinator.movin_collectives.control_group
        )
        assert coordinator._cpu_group_generation == cycle + 1
        dist.barrier()

    assert_cpu_collective(coordinator, 4)
    coordinator.suspend_cpu_group()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
