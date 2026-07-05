from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

import sglang.srt.distributed.parallel_state as parallel_state
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _AllGather:
    def __init__(self, supported: bool):
        self.supported = supported
        self.calls = 0

    def should_all_gather(self, _input, _output):
        return self.supported

    def all_gather(self, input_, output):
        self.calls += 1
        output.copy_(input_.repeat(2))


def _coordinator_with_all_gather(supported: bool) -> GroupCoordinator:
    coordinator = GroupCoordinator.__new__(GroupCoordinator)
    coordinator.ag_comm = _AllGather(supported)
    coordinator.ca_comm = None
    coordinator.pynccl_comm = None
    return coordinator


def test_checkpointable_all_gather_routes_without_external_communicator():
    coordinator = _coordinator_with_all_gather(supported=True)
    input_ = torch.arange(4)
    output = torch.empty(8, dtype=input_.dtype)

    coordinator._all_gather_into_tensor(output, input_)

    torch.testing.assert_close(output, input_.repeat(2))
    assert coordinator.ag_comm.calls == 1


def test_checkpointable_all_gather_fails_closed_for_unsupported_input():
    coordinator = _coordinator_with_all_gather(supported=False)

    with pytest.raises(RuntimeError, match="checkpointable all-gather"):
        coordinator._all_gather_into_tensor(torch.empty(8), torch.empty(4))


def test_checkpointable_group_disables_external_communicators(monkeypatch):
    constructor = Mock(return_value=object())
    monkeypatch.setattr(parallel_state, "GroupCoordinator", constructor)

    parallel_state.init_model_parallel_group(
        [[0, 1]],
        local_rank=0,
        backend="nccl",
        use_checkpointable_collectives=True,
    )

    kwargs = constructor.call_args.kwargs
    assert kwargs["use_pynccl"] is False
    assert kwargs["use_pymscclpp"] is False
    assert kwargs["use_custom_allreduce"] is True
    assert kwargs["use_torch_symm_mem_all_reduce"] is False
    assert kwargs["use_checkpointable_collectives"] is True


def test_checkpointable_group_rejects_explicit_external_communicator():
    with pytest.raises(ValueError, match="external communicators"):
        parallel_state.init_model_parallel_group(
            [[0, 1]],
            local_rank=0,
            backend="nccl",
            use_pynccl=True,
            use_checkpointable_collectives=True,
        )
