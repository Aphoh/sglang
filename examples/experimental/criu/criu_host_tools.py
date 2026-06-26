"""Host-side process and GPU inspection helpers for the CRIU runner."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def parse_compute_apps(rows: Iterable[str]) -> dict[int, set[str]]:
    residency: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        gpu_uuid, raw_pid = (value.strip() for value in row.split(",", 1))
        residency[int(raw_pid)].add(gpu_uuid)
    return dict(residency)


def query_compute_apps() -> dict[int, set[str]]:
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    return parse_compute_apps(rows)


def process_start_time(pid: int) -> int:
    return parse_process_start_time(Path(f"/proc/{pid}/stat").read_text())


def parse_process_start_time(stat: str) -> int:
    return int(stat.rsplit(")", 1)[1].split()[19])


def pinned_movin_commit(pyproject_path: Path) -> str:
    project = tomllib.loads(pyproject_path.read_text())
    dependencies = project["project"]["optional-dependencies"]["criu"]
    matches = [
        re.search(r"warnold-movin\.git@([0-9a-f]{40})$", dependency)
        for dependency in dependencies
    ]
    commits = [match.group(1) for match in matches if match is not None]
    if len(commits) != 1:
        raise ValueError(
            f"expected exactly one immutable Movin dependency in {pyproject_path}"
        )
    return commits[0]


def process_tree(root: int, workers: Iterable[int]) -> list[int]:
    children: dict[int, list[int]] = defaultdict(list)
    for status_path in Path("/proc").glob("[0-9]*/status"):
        try:
            fields = dict(
                line.partition(":")[::2]
                for line in status_path.read_text().splitlines()
                if ":" in line
            )
            children[int(fields["PPid"].strip())].append(
                int(fields["Pid"].strip())
            )
        except (OSError, KeyError, ValueError):
            continue
    for child_pids in children.values():
        child_pids.sort()

    ordered = [root]
    seen = {root}

    def append_subtree(pid: int) -> None:
        if pid in seen:
            return
        seen.add(pid)
        ordered.append(pid)
        for child in children.get(pid, ()):
            append_subtree(child)

    for worker in workers:
        append_subtree(worker)
    for child in children.get(root, ()):
        append_subtree(child)
    return ordered


def record_gpu_residency(
    phase: str,
    worker_path: Path,
    expected_gpu_uuids: set[str],
    output_path: Path,
) -> None:
    workers = {int(pid) for pid in json.loads(worker_path.read_text())}
    observed_by_pid = query_compute_apps()
    residency = {pid: observed_by_pid.get(pid, set()) for pid in workers}
    cpu_only_workers = sorted(pid for pid, gpu_uuids in residency.items() if not gpu_uuids)
    observed = set().union(*residency.values()) if residency else set()
    if observed != expected_gpu_uuids:
        formatted = {pid: sorted(values) for pid, values in sorted(residency.items())}
        raise RuntimeError(
            f"{phase} GPU residency mismatch: expected={sorted(expected_gpu_uuids)}, "
            f"observed={sorted(observed)}, cpu_only_workers={cpu_only_workers}, "
            f"workers={formatted}"
        )

    payload = json.loads(output_path.read_text()) if output_path.exists() else {}
    payload[phase] = {
        "cpu_only_worker_pids": cpu_only_workers,
        "expected_gpu_uuids": sorted(expected_gpu_uuids),
        "worker_gpu_uuids": {
            str(pid): sorted(gpu_uuids)
            for pid, gpu_uuids in sorted(residency.items())
        },
    }
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_path)
    print(json.dumps({phase: payload[phase]}, sort_keys=True), flush=True)


def _read_pids(path: Path) -> list[int]:
    return [int(pid) for pid in json.loads(path.read_text())]


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_time_parser = subparsers.add_parser("process-start-time")
    start_time_parser.add_argument("pid", type=int)

    tree_parser = subparsers.add_parser("process-tree")
    tree_parser.add_argument("root", type=int)
    tree_parser.add_argument("worker_path", type=Path)

    residency_parser = subparsers.add_parser("record-gpu-residency")
    residency_parser.add_argument("phase")
    residency_parser.add_argument("worker_path", type=Path)
    residency_parser.add_argument("expected_gpu_uuids")
    residency_parser.add_argument("output_path", type=Path)

    migration_parser = subparsers.add_parser("migration-pids")
    migration_parser.add_argument("source_gpu_uuids")
    migration_parser.add_argument("pids", nargs="+", type=int)

    pin_parser = subparsers.add_parser("pinned-movin-commit")
    pin_parser.add_argument("pyproject_path", type=Path)

    args = parser.parse_args()
    if args.command == "process-start-time":
        print(process_start_time(args.pid))
    elif args.command == "process-tree":
        for pid in process_tree(args.root, _read_pids(args.worker_path)):
            print(pid)
    elif args.command == "record-gpu-residency":
        record_gpu_residency(
            args.phase,
            args.worker_path,
            set(filter(None, args.expected_gpu_uuids.split(","))),
            args.output_path,
        )
    elif args.command == "migration-pids":
        source_gpu_uuids = set(filter(None, args.source_gpu_uuids.split(",")))
        residency = query_compute_apps()
        for pid in sorted(args.pids):
            if residency.get(pid, set()) & source_gpu_uuids:
                print(pid)
    elif args.command == "pinned-movin-commit":
        print(pinned_movin_commit(args.pyproject_path))


if __name__ == "__main__":
    main()
