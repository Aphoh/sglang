import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

MODULE_PATH = (
    Path(__file__).parents[4]
    / "python/sglang/srt/distributed/device_communicators/movin_nixl.py"
)
SPEC = importlib.util.spec_from_file_location("sglang_movin_nixl_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
MovinCollectiveConfig = MODULE.MovinCollectiveConfig
create_movin_collectives = MODULE.create_movin_collectives


class FakeBackend:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeManager:
    def __init__(self, *, all_reduce=None, all_gather=None, participants=None):
        self.all_reduce_backend = all_reduce
        self.all_gather_backend = all_gather
        self.participants = participants or {}


def test_disabled_config_does_not_import_movin(monkeypatch):
    monkeypatch.setitem(sys.modules, "movin", None)
    config = MovinCollectiveConfig(
        all_reduce_backend=None,
        enable_all_gather=False,
        all_reduce_max_elems=1,
        all_gather_max_elems=1,
        checkpointable=False,
        force_nixl_after_restore=False,
        restore_probe=False,
    )

    assert (
        create_movin_collectives(
            object(),
            torch.device("cpu"),
            "tp",
            config,
        )
        is None
    )


def test_create_movin_collectives_uses_unified_manager(monkeypatch):
    fake_movin = SimpleNamespace(
        CollectiveManager=FakeManager,
        TorchDistributedNixlAllReduce=FakeBackend,
    )
    monkeypatch.setitem(sys.modules, "movin", fake_movin)
    monkeypatch.setattr(MODULE, "FlashInferSymmetricAllGather", FakeBackend)
    checkpoint_participants = {"flashinfer_attention": object()}
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.flashinfer_comm_fusion",
        SimpleNamespace(
            get_flashinfer_checkpoint_participants=(
                lambda group_name: checkpoint_participants
            )
        ),
    )
    group = object()
    device = torch.device("cuda", 3)
    config = MovinCollectiveConfig(
        all_reduce_backend="movin_nixl",
        enable_all_gather=True,
        all_reduce_max_elems=131072,
        all_gather_max_elems=262144,
        checkpointable=True,
        force_nixl_after_restore=True,
        restore_probe=True,
    )

    manager = create_movin_collectives(group, device, "tp", config)

    assert isinstance(manager, FakeManager)
    assert manager.all_reduce_backend.kwargs == {
        "group": group,
        "device": device,
        "group_name": "tp",
        "max_elems": 131072,
        "checkpointable": True,
        "force_nixl_after_restore": True,
        "restore_probe": True,
    }
    assert manager.all_gather_backend.kwargs == {
        "group": group,
        "device": device,
        "group_name": "tp",
        "max_elems": 262144,
        "restore_probe": True,
    }
    assert manager.participants is checkpoint_participants


def test_checkpoint_only_config_still_creates_lifecycle_manager(monkeypatch):
    fake_movin = SimpleNamespace(
        CollectiveManager=FakeManager,
        TorchDistributedNixlAllReduce=FakeBackend,
    )
    participant = object()
    monkeypatch.setitem(sys.modules, "movin", fake_movin)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.flashinfer_comm_fusion",
        SimpleNamespace(
            get_flashinfer_checkpoint_participants=lambda group_name: {
                "workspace": participant
            }
        ),
    )
    config = MovinCollectiveConfig(
        all_reduce_backend=None,
        enable_all_gather=False,
        all_reduce_max_elems=1,
        all_gather_max_elems=1,
        checkpointable=True,
        force_nixl_after_restore=False,
        restore_probe=False,
    )

    manager = create_movin_collectives(
        object(),
        torch.device("cpu"),
        "tp",
        config,
    )

    assert isinstance(manager, FakeManager)
    assert manager.all_reduce_backend is None
    assert manager.all_gather_backend is None
    assert manager.participants == {"workspace": participant}


def test_create_movin_collectives_ignores_non_tp_groups(monkeypatch):
    monkeypatch.setitem(sys.modules, "movin", SimpleNamespace())
    config = MovinCollectiveConfig(
        all_reduce_backend="movin_nixl",
        enable_all_gather=True,
        all_reduce_max_elems=1,
        all_gather_max_elems=1,
        checkpointable=True,
        force_nixl_after_restore=False,
        restore_probe=False,
    )

    assert create_movin_collectives(object(), torch.device("cpu"), "pp", config) is None
