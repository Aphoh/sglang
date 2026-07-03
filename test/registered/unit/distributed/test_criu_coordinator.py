import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch


class EnvValue:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def load_coordinator_module():
    envs = SimpleNamespace(
        SGLANG_CRIU_DEVICE_STORE=EnvValue(None),
        SGLANG_CRIU_SUSPEND_DEVICE_PROCESS_GROUP=EnvValue(False),
    )
    environ_module = ModuleType("sglang.srt.environ")
    environ_module.envs = envs
    io_module = ModuleType("sglang.srt.managers.io_struct")
    io_module.RpcReqInput = type("RpcReqInput", (), {})
    zmq_module = ModuleType("zmq")
    zmq_module.NOBLOCK = 1
    zmq_module.ZMQError = type("ZMQError", (Exception,), {})
    zmq_module.Again = type("Again", (zmq_module.ZMQError,), {})
    stubs = {
        "sglang.srt.environ": environ_module,
        "sglang.srt.managers.io_struct": io_module,
        "zmq": zmq_module,
    }
    module_path = (
        Path(__file__).parents[4]
        / "python/sglang/srt/distributed/criu_coordinator.py"
    )
    spec = importlib.util.spec_from_file_location(
        "sglang_criu_coordinator_test",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module, envs


MODULE, ENVS = load_coordinator_module()


class FakeManager:
    def status(self):
        return {}


class FakeGroup:
    def __init__(self, manager):
        self.checkpoint_collectives = manager


def make_coordinator(monkeypatch, groups):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    return MODULE.CriuCheckpointCoordinator(
        device=torch.device("cpu"),
        groups=groups,
        server_args=object(),
        model_config=object(),
    )


def test_collective_sets_deduplicate_shared_manager(monkeypatch):
    manager = FakeManager()
    coordinator = make_coordinator(
        monkeypatch,
        [FakeGroup(manager), FakeGroup(manager)],
    )

    assert list(coordinator._collective_sets()) == [manager]


def test_checkpoint_mode_requires_device_store_at_startup(monkeypatch):
    ENVS.SGLANG_CRIU_SUSPEND_DEVICE_PROCESS_GROUP.value = True
    ENVS.SGLANG_CRIU_DEVICE_STORE.value = None
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    try:
        with pytest.raises(RuntimeError, match="SGLANG_CRIU_DEVICE_STORE"):
            MODULE.CriuCheckpointCoordinator(
                device=torch.device("cpu"),
                groups=[],
                server_args=object(),
                model_config=object(),
            )
    finally:
        ENVS.SGLANG_CRIU_SUSPEND_DEVICE_PROCESS_GROUP.value = False


def test_file_consensus_cleans_single_rank_artifacts(tmp_path, monkeypatch):
    ENVS.SGLANG_CRIU_DEVICE_STORE.value = str(tmp_path / "store")
    coordinator = make_coordinator(monkeypatch, [])

    assert coordinator._converge_file_status(
        namespace="unit.success",
        success=True,
        error="",
        timeout=1,
    ) == (True, "")
    assert list(tmp_path.iterdir()) == []


def test_run_phase_converges_local_failure(tmp_path, monkeypatch):
    ENVS.SGLANG_CRIU_DEVICE_STORE.value = str(tmp_path / "store")
    coordinator = make_coordinator(monkeypatch, [])

    def fail():
        raise ValueError("boom")

    with pytest.raises(RuntimeError, match="rank 0: boom"):
        coordinator._run_phase("unit-failure", fail, timeout=1)


@pytest.mark.parametrize(
    ("method", "success", "expected"),
    [
        ("ordinary_rpc", True, True),
        ("ordinary_rpc", False, True),
        (MODULE.CriuCheckpointCoordinator.PREPARE_RPC, True, False),
        (MODULE.CriuCheckpointCoordinator.PREPARE_RPC, False, False),
        (MODULE.CriuCheckpointCoordinator.RESTORE_RPC, True, True),
        (MODULE.CriuCheckpointCoordinator.RESTORE_RPC, False, False),
    ],
)
def test_rpc_barrier_policy_preserves_ordinary_rpc_barrier(
    method,
    success,
    expected,
):
    coordinator = MODULE.CriuCheckpointCoordinator.__new__(
        MODULE.CriuCheckpointCoordinator
    )
    assert coordinator.should_barrier_after_rpc(method, success=success) is expected


def test_restore_polling_ignores_only_would_block(tmp_path):
    ENVS.SGLANG_CRIU_DEVICE_STORE.value = str(tmp_path / "store")
    tp_group = SimpleNamespace(cpu_group_generation=0, is_first_rank=True)
    socket = SimpleNamespace(
        recv_pyobj=lambda flags: (_ for _ in ()).throw(MODULE.zmq.Again())
    )

    assert MODULE.receive_restore_request(
        tp_group=tp_group,
        recv_from_rpc=socket,
    ) == []


def test_restore_polling_propagates_broken_socket(tmp_path):
    ENVS.SGLANG_CRIU_DEVICE_STORE.value = str(tmp_path / "store")
    tp_group = SimpleNamespace(cpu_group_generation=0, is_first_rank=True)
    socket = SimpleNamespace(
        recv_pyobj=lambda flags: (_ for _ in ()).throw(MODULE.zmq.ZMQError())
    )

    with pytest.raises(MODULE.zmq.ZMQError):
        MODULE.receive_restore_request(
            tp_group=tp_group,
            recv_from_rpc=socket,
        )
