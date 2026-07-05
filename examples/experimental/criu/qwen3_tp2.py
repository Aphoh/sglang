"""Deterministic Qwen3 TP2 workload around an external CUDA/CRIU checkpoint."""

import argparse
import json
import os
import time
from collections.abc import Iterable
from pathlib import Path


def wait_for(path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.1)


def assert_no_io_uring(pids: Iterable[int]) -> None:
    rings: list[str] = []
    for pid in pids:
        for fd in Path(f"/proc/{pid}/fd").iterdir():
            try:
                if os.readlink(fd) == "anon_inode:[io_uring]":
                    rings.append(f"{pid}/{fd.name}")
            except FileNotFoundError:
                pass
    if rings:
        raise RuntimeError(f"CRIU-incompatible io_uring descriptors remain: {rings}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--rendezvous", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--mem-fraction-static", type=float, default=0.1)
    parser.add_argument("--max-total-tokens", type=int, default=4096)
    args = parser.parse_args()

    args.rendezvous.mkdir(parents=True, exist_ok=True)
    for name in ("job-ready", "resume", "passed", "result.json"):
        (args.rendezvous / name).unlink(missing_ok=True)

    import sglang as sgl

    engine = sgl.Engine(
        model_path=args.model,
        tp_size=args.tp_size,
        context_length=2048,
        mem_fraction_static=args.mem_fraction_static,
        max_total_tokens=args.max_total_tokens,
        max_running_requests=4,
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        enable_criu_checkpoint=True,
        criu_store_prefix=str(args.rendezvous / "torch-store"),
        log_level="info",
    )
    try:
        prompt = "The capital of France is"
        sampling = {"temperature": 0, "max_new_tokens": 8}
        engine.generate(prompt, sampling)
        started = time.perf_counter()
        before = engine.generate(prompt, sampling)["text"]
        before_ms = (time.perf_counter() - started) * 1000
        time.sleep(0.5)

        started = time.perf_counter()
        engine.suspend_checkpoint()
        suspend_ms = (time.perf_counter() - started) * 1000
        child_pids = engine.get_all_child_pids()
        assert_no_io_uring([os.getpid(), *child_pids])
        (args.rendezvous / "controller-pid").write_text(f"{os.getpid()}\n")
        (args.rendezvous / "worker-pids.json").write_text(json.dumps(child_pids) + "\n")
        (args.rendezvous / "job-ready").touch()
        started = time.perf_counter()
        wait_for(args.rendezvous / "resume", args.timeout)
        external_checkpoint_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        engine.resume_checkpoint()
        resume_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        after = engine.generate(prompt, sampling)["text"]
        after_ms = (time.perf_counter() - started) * 1000
        if after != before:
            raise AssertionError(f"generation changed: {before!r} != {after!r}")

        result = {
            "after": after,
            "after_ms": after_ms,
            "before": before,
            "before_ms": before_ms,
            "external_checkpoint_ms": external_checkpoint_ms,
            "resume_ms": resume_ms,
            "suspend_ms": suspend_ms,
        }
        (args.rendezvous / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        (args.rendezvous / "passed").touch()
        print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
