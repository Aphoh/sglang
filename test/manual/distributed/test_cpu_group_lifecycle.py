"""Exercise repeated SGLang CPU process-group teardown and recreation.

Run with:
  torchrun --standalone --nproc-per-node=2 \
    test/manual/distributed/test_cpu_group_lifecycle.py
"""

import os
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.distributed as dist

from sglang.srt.distributed.criu_coordinator import (
    CheckpointState,
    CriuCheckpointCoordinator,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator


class FakeCheckpointCollectives:
    has_all_reduce = True
    has_all_gather = True

    def __init__(self, control_group):
        self.control_group = control_group
        self.detached = False

    def capture(self):
        return nullcontext()

    def prepare_checkpoint(self) -> None:
        assert self.control_group is not None
        self.detached = True

    def restore_after_checkpoint(self) -> None:
        assert self.control_group is not None
        assert self.detached
        self.detached = False

    def set_control_group(self, control_group) -> None:
        self.control_group = control_group

    def status(self):
        return {"fake": [0, 0, 0]}

    def close(self) -> None:
        self.control_group = None


def make_cpu_only_coordinator(local_rank: int) -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=[list(range(dist.get_world_size()))],
        local_rank=local_rank,
        torch_distributed_backend="gloo",
        use_pynccl=False,
        use_pymscclpp=False,
        use_custom_allreduce=False,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
        group_name="tp",
        device_group_override=dist.group.WORLD,
        create_device_group=False,
    )


def assert_cpu_collective(coordinator: GroupCoordinator, cycle: int) -> None:
    value = torch.tensor([dist.get_rank() + cycle], dtype=torch.int64)
    dist.all_reduce(value, group=coordinator.cpu_group)
    expected = sum(range(dist.get_world_size())) + cycle * dist.get_world_size()
    assert value.item() == expected


def make_checkpoint_coordinator(group: GroupCoordinator):
    server_args = SimpleNamespace(
        pp_size=1,
        dp_size=1,
        ep_size=1,
        moe_dp_size=1,
        attn_cp_size=1,
        enable_dp_attention=False,
        moe_a2a_backend="none",
        disaggregation_mode="null",
        enable_hierarchical_cache=False,
        hicache_storage_backend=None,
        enable_hisparse=False,
        disable_radix_cache=True,
        disable_custom_all_reduce=False,
        enable_flashinfer_allreduce_fusion=False,
        enable_symm_mem=False,
    )
    model_config = SimpleNamespace(hf_text_config=SimpleNamespace())
    return CriuCheckpointCoordinator(
        device=group.device,
        groups=[group],
        server_args=server_args,
        model_config=model_config,
    )


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    os.environ["SGLANG_CRIU_SUSPEND_DEVICE_PROCESS_GROUP"] = "1"
    os.environ["SGLANG_CRIU_DEVICE_STORE"] = (
        f"/tmp/sglang-cpu-lifecycle-{os.environ['MASTER_PORT']}"
    )

    coordinator = make_cpu_only_coordinator(local_rank)
    coordinator.checkpoint_collectives.close()
    coordinator.checkpoint_collectives = FakeCheckpointCollectives(
        coordinator.cpu_group
    )
    checkpoint = make_checkpoint_coordinator(coordinator)
    graph_value = torch.zeros(1, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_value.add_(1)

    for cycle in range(4):
        assert_cpu_collective(coordinator, cycle)
        graph.replay()
        torch.cuda.synchronize()
        assert graph_value.item() == cycle + 1

        checkpoint.prepare()
        assert checkpoint.state is CheckpointState.PREPARED
        assert coordinator.cpu_group is None
        assert not dist.is_initialized()
        assert coordinator.device_group is None

        checkpoint.restore()
        assert checkpoint.state is CheckpointState.READY
        assert dist.is_initialized()
        assert coordinator.device_group is dist.group.WORLD
        assert coordinator.cpu_group_generation == cycle + 1
        assert (
            coordinator.checkpoint_collectives.control_group is coordinator.cpu_group
        )
        assert not coordinator.checkpoint_collectives.detached
        dist.barrier()

    assert_cpu_collective(coordinator, 4)
    coordinator.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
