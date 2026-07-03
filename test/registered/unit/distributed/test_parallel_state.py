"""
Test file to verify the correctness of parallel group calculations.

This test validates that the parallel group initialization creates the correct
groups for different parallelism configurations including:
- Tensor parallelism (TP)
- Pipeline parallelism (PP)
- Attention context parallelism (attn_cp)
- Attention data parallelism (attn_dp)
- MoE expert parallelism (EP)
- MoE data parallelism (moe_dp)

These tests call the ACTUAL initialize_model_parallel() function with mocked
distributed backend to verify the group construction logic.

## How These Tests Work

initialize_model_parallel() creates ALL groups for ALL ranks in a single call.
For example, when creating TP groups with tp_size=2 and world_size=8:

    group_ranks = [[0,1], [2,3], [4,5], [6,7]]  # ALL groups created
    _TP = init_model_parallel_group(group_ranks, local_rank, ...)

ALL ranks call this function and get the same complete group structure. Each rank
then figures out which specific group(s) it belongs to.

Our tests:
1. Mock the distributed backend (no real GPUs needed)
2. Mock init_model_parallel_group to capture the group_ranks parameter
3. Call the real initialize_model_parallel()
4. Verify group_ranks contains the expected complete group structure

We only need to simulate rank 0 because we're testing the group creation logic,
not the per-rank group membership logic.
"""

from __future__ import annotations

import sys
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)
from sglang.test.test_utils import run_distributed_test

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=8, suite="stage-b-test-1-gpu-small-amd")
register_cpu_ci(est_time=8, suite="base-a-test-cpu")

# Import the actual parallel_state module
parallel_state = pytest.importorskip("sglang.srt.distributed.parallel_state")


def test_parallel_group_construction_tp8_attn_cp2():
    """
    Test parallel group construction for 8 GPU configuration with:
    - tensor_model_parallel_size = 8
    - attention_context_model_parallel_size = 2

    Expected groups based on docstring example:
        1 tensor model-parallel group:
            [g0, g1, g2, g3, g4, g5, g6, g7]
        4 attention context-parallel groups:
            [g0, g4], [g1, g5], [g2, g6], [g3, g7]

    This test calls the ACTUAL initialize_model_parallel() and verifies the groups.

    Note: We simulate only rank 0 here, but initialize_model_parallel() creates
    ALL groups for ALL ranks in a single call. We capture these groups via mocking
    and verify the complete group structure.
    """
    world_size = 8

    # Mock the distributed backend
    # Note: get_rank() returns 0 because we're testing from a single process,
    # but initialize_model_parallel() still creates all groups for all ranks
    with (
        patch.object(parallel_state, "_WORLD", None),
        patch.object(parallel_state, "_TP", None),
        patch.object(parallel_state, "_ATTN_CP", None),
        patch.object(parallel_state, "_ATTN_TP", None),
        patch.object(parallel_state, "_PP", None),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=world_size),
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.get_backend", return_value="nccl"),
    ):

        # Mock init_model_parallel_group to capture the groups being created
        created_groups = {}

        def mock_init_model_parallel_group(group_ranks, local_rank, backend, **kwargs):
            group_name = kwargs.get("group_name", "unknown")
            created_groups[group_name] = group_ranks

            # Create a mock group object
            mock_group = Mock()
            mock_group.device_group = Mock()
            return mock_group

        with (
            patch.object(
                parallel_state,
                "init_model_parallel_group",
                side_effect=mock_init_model_parallel_group,
            ),
            patch.object(parallel_state, "get_world_group") as mock_world_group,
        ):

            # Mock world group
            mock_world = Mock()
            mock_world.device_group = Mock()
            mock_world.local_rank = 0
            mock_world_group.return_value = mock_world

            # Call the actual function
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=8,
                pipeline_model_parallel_size=1,
                attention_context_model_parallel_size=2,
            )

            # Verify TP groups
            tp_groups = created_groups.get("tp", [])
            assert len(tp_groups) == 1, f"Expected 1 TP group, got {len(tp_groups)}"
            assert tp_groups[0] == [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
            ], f"Wrong TP group: {tp_groups[0]}"

            # Verify ATTN_CP groups
            attn_cp_groups = created_groups.get("attn_cp", [])
            assert (
                len(attn_cp_groups) == 4
            ), f"Expected 4 ATTN_CP groups, got {len(attn_cp_groups)}"
            expected_attn_cp = [
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ]
            assert (
                attn_cp_groups == expected_attn_cp
            ), f"Wrong ATTN_CP groups: {attn_cp_groups}"

            print("TP=8, Attn CP=2 group construction verified")

            # Cleanup
            parallel_state.destroy_model_parallel()


def test_parallel_group_construction_tp8_moe_ep4_cp2():
    """
    Test parallel group construction for 8 GPU configuration with:
    - tensor_model_parallel_size = 8
    - expert_model_parallel_size = 4
    - moe_data_model_parallel_size = 2

    Expected groups:
        1 tensor model-parallel group:
            [g0, g1, g2, g3, g4, g5, g6, g7]
        2 MoE expert-parallel groups:
            [g0, g1, g2, g3], [g4, g5, g6, g7]
        4 MoE data-parallel groups:
            [g0, g4], [g1, g5], [g2, g6], [g3, g7]
    """
    world_size = 8

    # Mock the distributed backend
    with (
        patch.object(parallel_state, "_WORLD", None),
        patch.object(parallel_state, "_TP", None),
        patch.object(parallel_state, "_MOE_EP", None),
        patch.object(parallel_state, "_MOE_DP", None),
        patch.object(parallel_state, "_MOE_TP", None),
        patch.object(parallel_state, "_PP", None),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=world_size),
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.get_backend", return_value="nccl"),
    ):

        # Mock init_model_parallel_group to capture the groups being created
        created_groups = {}

        def mock_init_model_parallel_group(group_ranks, local_rank, backend, **kwargs):
            group_name = kwargs.get("group_name", "unknown")
            created_groups[group_name] = group_ranks

            # Create a mock group object
            mock_group = Mock()
            mock_group.device_group = Mock()
            return mock_group

        with (
            patch.object(
                parallel_state,
                "init_model_parallel_group",
                side_effect=mock_init_model_parallel_group,
            ),
            patch.object(parallel_state, "get_world_group") as mock_world_group,
        ):

            # Mock world group
            mock_world = Mock()
            mock_world.device_group = Mock()
            mock_world.local_rank = 0
            mock_world_group.return_value = mock_world

            # Call the actual function
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=8,
                expert_model_parallel_size=4,
                pipeline_model_parallel_size=1,
                moe_data_model_parallel_size=2,
            )

            # Verify TP groups
            tp_groups = created_groups.get("tp", [])
            assert len(tp_groups) == 1, f"Expected 1 TP group, got {len(tp_groups)}"
            assert tp_groups[0] == [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
            ], f"Wrong TP group: {tp_groups[0]}"

            # Verify MOE_EP groups
            moe_ep_groups = created_groups.get("moe_ep", [])
            assert (
                len(moe_ep_groups) == 2
            ), f"Expected 2 MOE_EP groups, got {len(moe_ep_groups)}"
            expected_moe_ep = [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
            ]
            assert (
                moe_ep_groups == expected_moe_ep
            ), f"Wrong MOE_EP groups: {moe_ep_groups}"

            # Verify MOE_DP groups
            moe_dp_groups = created_groups.get("moe_dp", [])
            assert (
                len(moe_dp_groups) == 4
            ), f"Expected 4 MOE_DP groups, got {len(moe_dp_groups)}"
            expected_moe_dp = [
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ]
            assert (
                moe_dp_groups == expected_moe_dp
            ), f"Wrong MOE_DP groups: {moe_dp_groups}"

            print("TP=8, MoE EP=4, MoE CP=2 group construction verified")

            # Cleanup
            parallel_state.destroy_model_parallel()


def _make_coordinator_stub(
    binding,
    *,
    cpu_group=None,
    owned_device_group=None,
):
    coordinator = parallel_state.GroupCoordinator.__new__(
        parallel_state.GroupCoordinator
    )
    coordinator.unique_name = "test:0"
    coordinator.device_group_binding = binding
    coordinator._cpu_group = cpu_group
    coordinator._owned_device_group = owned_device_group
    coordinator._destroyed = False
    return coordinator


def test_device_group_bindings_resolve_current_group():
    first_cpu_group = object()
    second_cpu_group = object()
    owned_device_group = object()

    cpu_alias = _make_coordinator_stub(
        parallel_state.DeviceGroupBinding.CPU_ALIAS,
        cpu_group=first_cpu_group,
    )
    assert cpu_alias.device_group is first_cpu_group
    cpu_alias._cpu_group = second_cpu_group
    assert cpu_alias.device_group is second_cpu_group

    owned = _make_coordinator_stub(
        parallel_state.DeviceGroupBinding.OWNED,
        owned_device_group=owned_device_group,
    )
    assert owned.device_group is owned_device_group


def test_world_binding_resolves_reinitialized_default_group():
    coordinator = _make_coordinator_stub(parallel_state.DeviceGroupBinding.WORLD)
    first_world = object()
    second_world = object()
    world = torch.distributed.distributed_c10d._world

    with patch.object(world, "_default_pg", first_world):
        assert coordinator.device_group is first_world
    with patch.object(world, "_default_pg", second_world):
        assert coordinator.device_group is second_world

    assert not coordinator.has_device_group
    with pytest.raises(RuntimeError, match="unavailable"):
        _ = coordinator.device_group


def test_cpu_group_release_and_renew(monkeypatch):
    initial_group = object()
    renewed_group = object()
    active_ranks = object()
    coordinator = _make_coordinator_stub(
        parallel_state.DeviceGroupBinding.CPU_ALIAS,
        cpu_group=initial_group,
    )
    coordinator.rank = 0
    coordinator._cpu_group_generation = 0
    coordinator._cpu_group_spec = Mock()
    coordinator._cpu_group_spec.create_for_rank.return_value = (
        renewed_group,
        active_ranks,
    )
    coordinator.use_message_queue_broadcaster = False
    coordinator.world_size = 2
    coordinator.mq_broadcaster = None

    destroy_process_group = Mock()
    monkeypatch.setattr(
        torch.distributed,
        "destroy_process_group",
        destroy_process_group,
    )

    coordinator.release_cpu_group()
    destroy_process_group.assert_called_once_with(initial_group)
    assert not coordinator.has_cpu_group
    assert not coordinator.has_device_group
    with pytest.raises(RuntimeError, match="unavailable"):
        _ = coordinator.cpu_group

    assert coordinator.renew_cpu_group() is renewed_group
    assert coordinator.cpu_group is renewed_group
    assert coordinator.device_group is renewed_group
    assert coordinator.active_ranks_cpu is active_ranks
    assert coordinator.cpu_group_generation == 1

    assert coordinator.renew_cpu_group() is renewed_group
    coordinator._cpu_group_spec.create_for_rank.assert_called_once_with(0)


@pytest.mark.parametrize(
    "communicator_name",
    (
        "pynccl_comm",
        "pymscclpp_comm",
        "ca_comm",
        "qr_comm",
        "torch_symm_mem_comm",
        "hpu_communicator",
        "xpu_communicator",
        "npu_communicator",
    ),
)
def test_cpu_group_release_rejects_retained_communicator(
    monkeypatch,
    communicator_name,
):
    cpu_group = object()
    coordinator = _make_coordinator_stub(
        parallel_state.DeviceGroupBinding.CPU_ALIAS,
        cpu_group=cpu_group,
    )
    setattr(coordinator, communicator_name, Mock(group=cpu_group))

    destroy_process_group = Mock()
    monkeypatch.setattr(
        torch.distributed,
        "destroy_process_group",
        destroy_process_group,
    )

    with pytest.raises(RuntimeError, match=communicator_name):
        coordinator.release_cpu_group()
    assert coordinator.cpu_group is cpu_group
    destroy_process_group.assert_not_called()


def _run_cpu_group_renewal(rank):
    # Exercise only Gloo lifecycle behavior, including when this test runs on
    # CUDA CI hosts where SGLang's platform detection selects a GPU.
    parallel_state.is_cuda_alike = lambda: False
    parallel_state._is_npu = False
    parallel_state._is_xpu = False
    parallel_state._is_musa = False

    group = parallel_state.GroupCoordinator(
        group_ranks=[list(range(torch.distributed.get_world_size()))],
        local_rank=rank,
        torch_distributed_backend="gloo",
        use_pynccl=False,
        use_pymscclpp=False,
        use_custom_allreduce=False,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
        group_name="renewable_test",
        device_group_binding=parallel_state.DeviceGroupBinding.CPU_ALIAS,
    )

    for generation in range(1, 5):
        value = torch.tensor([rank], dtype=torch.int64)
        torch.distributed.all_reduce(value, group=group.cpu_group)
        assert value.item() == sum(range(torch.distributed.get_world_size()))

        previous_group = group.cpu_group
        group.release_cpu_group()
        assert not group.has_cpu_group
        assert not group.has_device_group

        torch.distributed.barrier()
        group.renew_cpu_group()
        assert group.cpu_group is not previous_group
        assert group.device_group is group.cpu_group
        assert group.cpu_group_generation == generation
        torch.distributed.barrier()

    group.destroy()


def test_cpu_group_renews_across_processes():
    run_distributed_test(
        _run_cpu_group_renewal,
        world_size=2,
        backend="gloo",
    )


if __name__ == "__main__":
    # Run tests without requiring GPUs
    import sys

    try:
        test_parallel_group_construction_tp8_attn_cp2()
        test_parallel_group_construction_tp8_moe_ep4_cp2()

        sys.exit(0)
    except AssertionError as e:
        print(f"\n Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
