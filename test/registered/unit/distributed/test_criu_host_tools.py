import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[4]
    / "examples/experimental/criu/criu_host_tools.py"
)
SPEC = importlib.util.spec_from_file_location("sglang_criu_host_tools_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_compute_apps_groups_gpu_uuids_by_pid():
    assert MODULE.parse_compute_apps(
        [
            "GPU-a, 100",
            "GPU-b, 200",
            "GPU-c, 100",
        ]
    ) == {
        100: {"GPU-a", "GPU-c"},
        200: {"GPU-b"},
    }


def test_record_gpu_residency_is_atomic_and_preserves_phases(tmp_path, monkeypatch):
    worker_path = tmp_path / "workers.json"
    output_path = tmp_path / "residency.json"
    worker_path.write_text("[100, 200]\n")
    monkeypatch.setattr(
        MODULE,
        "query_compute_apps",
        lambda: {100: {"GPU-a"}, 200: {"GPU-b"}},
    )

    MODULE.record_gpu_residency(
        "before",
        worker_path,
        {"GPU-a", "GPU-b"},
        output_path,
    )
    MODULE.record_gpu_residency(
        "after",
        worker_path,
        {"GPU-a", "GPU-b"},
        output_path,
    )

    result = json.loads(output_path.read_text())
    assert set(result) == {"before", "after"}
    assert result["after"]["worker_gpu_uuids"] == {
        "100": ["GPU-a"],
        "200": ["GPU-b"],
    }


def test_record_gpu_residency_rejects_partial_migration(tmp_path, monkeypatch):
    worker_path = tmp_path / "workers.json"
    worker_path.write_text("[100, 200]\n")
    monkeypatch.setattr(MODULE, "query_compute_apps", lambda: {100: {"GPU-a"}})

    with pytest.raises(RuntimeError, match="GPU residency mismatch"):
        MODULE.record_gpu_residency(
            "after",
            worker_path,
            {"GPU-a", "GPU-b"},
            tmp_path / "residency.json",
        )


def test_parse_process_start_time_handles_spaces_in_process_name():
    fields_after_name = ["S", *map(str, range(1, 22))]
    stat = f"123 (process with spaces) {' '.join(fields_after_name)}"

    assert MODULE.parse_process_start_time(stat) == 19
