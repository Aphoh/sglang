import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


def load_backend_module():
    utils_module = ModuleType("sglang.srt.utils")
    utils_module.is_flashinfer_available = lambda: False
    module_path = (
        Path(__file__).parents[4]
        / "python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        "sglang_flashinfer_comm_backend_test",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with patch.dict(sys.modules, {"sglang.srt.utils": utils_module}):
        spec.loader.exec_module(module)
    return module


MODULE = load_backend_module()


class FakeGroup:
    def rank(self):
        return 0

    def size(self):
        return 2


def test_broadcast_uses_group_relative_source(monkeypatch):
    calls = []

    def broadcast_object_list(values, **kwargs):
        calls.append((values, kwargs))

    monkeypatch.setattr(
        MODULE.dist,
        "broadcast_object_list",
        broadcast_object_list,
    )
    group = FakeGroup()
    backend = MODULE.TorchDistributedCommBackend(group)

    assert backend.bcast("value", root=1) == "value"
    assert calls == [
        (["value"], {"group": group, "group_src": 1}),
    ]
