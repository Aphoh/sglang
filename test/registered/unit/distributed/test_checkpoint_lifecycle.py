from __future__ import annotations

from types import SimpleNamespace

import pytest

import sglang.srt.distributed.checkpoint_lifecycle as lifecycle
from sglang.srt.distributed.checkpoint_lifecycle import (
    CheckpointLifecycle,
    CheckpointLifecycleError,
    CheckpointState,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Transaction:
    def __init__(self, name, calls, *, fail=None):
        self.name = name
        self.calls = calls
        self.fail = fail

    def preflight(self):
        self.calls.append(f"{self.name}.preflight")
        if self.fail == "preflight":
            raise RuntimeError("preflight failed")

    def suspend(self):
        self.calls.append(f"{self.name}.suspend")
        if self.fail == "suspend":
            raise RuntimeError("suspend failed")

    def resume(self):
        self.calls.append(f"{self.name}.resume")
        if self.fail == "resume":
            raise RuntimeError("resume failed")


def _lifecycle(monkeypatch, *, cpu_fail=None, device_fail=None):
    calls = []
    cpu = _Transaction("cpu", calls, fail=cpu_fail)
    device = _Transaction("device", calls, fail=device_fail)
    monkeypatch.setattr(lifecycle, "CpuGroupTransaction", lambda _bindings: cpu)
    monkeypatch.setattr(
        lifecycle,
        "DefaultDeviceGroupTransaction",
        lambda *args, **kwargs: device,
    )
    binding = object()
    groups = [
        SimpleNamespace(cpu_group_lifecycle=binding),
        SimpleNamespace(cpu_group_lifecycle=binding),
    ]
    return CheckpointLifecycle(groups, store_prefix="/tmp/checkpoint"), calls


def test_lifecycle_orders_suspend_and_restore(monkeypatch):
    checkpoint, calls = _lifecycle(monkeypatch)

    checkpoint.suspend()
    assert checkpoint.state is CheckpointState.SUSPENDED
    checkpoint.resume()

    assert checkpoint.state is CheckpointState.READY
    assert calls == [
        "device.preflight",
        "cpu.suspend",
        "device.suspend",
        "device.resume",
        "cpu.resume",
    ]


def test_preflight_failure_does_not_enter_mutating_state(monkeypatch):
    checkpoint, calls = _lifecycle(monkeypatch, device_fail="preflight")

    with pytest.raises(RuntimeError, match="preflight failed"):
        checkpoint.suspend()

    assert checkpoint.state is CheckpointState.READY
    assert calls == ["device.preflight"]


@pytest.mark.parametrize(
    ("cpu_fail", "device_fail", "operation"),
    [
        ("suspend", None, "suspend"),
        (None, "suspend", "suspend"),
        ("resume", None, "resume"),
        (None, "resume", "resume"),
    ],
)
def test_mutation_failure_is_terminal(monkeypatch, cpu_fail, device_fail, operation):
    checkpoint, _calls = _lifecycle(
        monkeypatch, cpu_fail=cpu_fail, device_fail=device_fail
    )
    if operation == "resume":
        checkpoint.suspend()

    with pytest.raises(CheckpointLifecycleError, match="worker restart required"):
        getattr(checkpoint, operation)()

    assert checkpoint.state is CheckpointState.FAILED
