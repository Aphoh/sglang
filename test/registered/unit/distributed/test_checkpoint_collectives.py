import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

MODULE_PATH = (
    Path(__file__).parents[4]
    / "python/sglang/srt/distributed/device_communicators/checkpoint_collectives.py"
)
SPEC = importlib.util.spec_from_file_location(
    "sglang_checkpoint_collectives_test", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeCollective:
    def __init__(self):
        self.disabled = False
        self.calls = []
        self.control_group = object()

    def prepare_checkpoint(self):
        self.calls.append("prepare")

    def restore_after_checkpoint(self):
        self.calls.append("restore")

    def set_control_group(self, group):
        self.control_group = group
        self.calls.append(("group", group))

    def status(self):
        return [0, 0, 0, 0]


class FakeAllGather(FakeCollective):
    def should_all_gather(self, input_, output=None):
        return True

    def all_gather(self, input_, output=None):
        self.calls.append("all_gather")
        return output

    def close(self):
        self.calls.append("close")


def test_manager_owns_native_lifecycle(monkeypatch):
    all_reduce = FakeCollective()
    all_gather = FakeAllGather()
    manager = MODULE.NativeCheckpointCollectives(all_reduce, all_gather)
    monkeypatch.setattr(MODULE.dist, "barrier", lambda group: None)

    manager.prepare_checkpoint()
    manager.restore_after_checkpoint()
    group = object()
    manager.set_control_group(group)
    manager.close()

    assert all_reduce.calls == ["prepare", "restore", ("group", group)]
    assert all_gather.calls == ["prepare", "restore", ("group", group), "close"]
    assert manager.status() == {
        "all_reduce": [0, 0, 0, 0],
        "all_gather": [0, 0, 0, 0],
    }


def test_factory_is_checkpoint_only(monkeypatch):
    monkeypatch.setattr(
        MODULE.envs,
        "SGLANG_CRIU_SUSPEND_DEVICE_PROCESS_GROUP",
        SimpleNamespace(get=lambda: False),
    )
    assert (
        MODULE.create_checkpoint_collectives(
            object(), torch.device("cpu"), "tp", FakeCollective()
        )
        is None
    )


def test_factory_ignores_non_tp_groups(monkeypatch):
    monkeypatch.setattr(
        MODULE.envs,
        "SGLANG_CRIU_SUSPEND_DEVICE_PROCESS_GROUP",
        SimpleNamespace(get=lambda: True),
    )
    assert (
        MODULE.create_checkpoint_collectives(
            object(), torch.device("cpu"), "pp", FakeCollective()
        )
        is None
    )
