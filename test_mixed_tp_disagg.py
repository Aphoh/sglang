#!/usr/bin/env python3
"""
Test script for mixed-TP disaggregated prefill/decode with Triton KV transfer.

Default Setup (3x GPUs):
- Prefill worker: TP=2 (GPU 0,1)
- Decode worker: TP=1 (GPU 2)

Reverse Setup (--reverse-tp):
- Prefill worker: TP=1 (GPU 0)
- Decode worker: TP=2 (GPU 1,2)

Decode Disagg Setup (--decode-disagg):
- Prefill worker: TP=2 (GPU 0,1)
- Decode Short worker: TP=4 (GPU 0,1,2,3) - handles short sequences
- Decode Long worker: TP=4 (GPU 4,5,6,7) - handles long sequences via migration

This tests the new Triton-based KV transfer method which:
1. Uses gather_kv to consolidate scattered KV data into a pinned CPU buffer
2. Transfers the contiguous buffer via NIXL (O(1) descriptors vs O(tokens × layers))
3. Uses scatter_kv to distribute data to the decode worker's KV cache

Usage:
    python test_mixed_tp_disagg.py --method triton
    python test_mixed_tp_disagg.py --method legacy
    python test_mixed_tp_disagg.py --method triton --reverse-tp
    python test_mixed_tp_disagg.py --method triton --decode-disagg
"""

import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

# Configuration
DEFAULT_MODEL_PATH = Path.home() / "proj/models/qwen3-4b"
HEALTH_URL = "http://localhost:8080/health"
CHAT_URL = "http://localhost:8080/v1/chat/completions"

# Decode disagg tier boundaries (token count thresholds)
DECODE_DISAGG_TIERS = [512, 4096]  # Short tier <= 512, Long tier <= 4096

# Event to signal all processes should stop
stop_event = threading.Event()

# Track processes with names for monitoring
processes: dict[str, subprocess.Popen] = {}

# ANSI colors for log prefixes
COLORS = {
    "prefill": "\033[32m",       # green
    "decode": "\033[36m",        # cyan
    "decode_short": "\033[36m",  # cyan
    "decode_long": "\033[35m",   # magenta
    "frontend": "\033[33m",      # yellow
    "reset": "\033[0m",
}

# Use dynamo venv
PYTHON_EXE = str(Path.home() / "proj/dynamo/.venv/bin/python")


def cleanup_stale_processes():
    """Kill any leftover sglang/dynamo processes from previous runs."""
    import signal

    print("🧹 Cleaning up stale processes...")

    # Find and kill processes matching patterns
    patterns = ["sglang.launch_server", "dynamo.sglang", "dynamo.frontend"]

    for pattern in patterns:
        try:
            # Use pgrep to find PIDs
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                        print(f"   Killed PID {pid} ({pattern})")
                    except (ProcessLookupError, ValueError):
                        pass
        except Exception:
            pass

    # Also check for processes on our ports
    for port in [8080, 10000, 11500]:
        try:
            result = subprocess.run(
                ["lsof", "-t", f"-i:{port}"],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                        print(f"   Killed PID {pid} (port {port})")
                    except (ProcessLookupError, ValueError):
                        pass
        except Exception:
            pass

    # Give processes time to die
    time.sleep(2)
    print("   Done!")


def get_common_args(model_path: Path, debug: bool = False, mem_fraction: float = 0.4) -> list[str]:
    """Get common arguments for sglang workers."""
    return [
        "--model-path", str(model_path),
        "--served-model-name", "model",
        "--context-length", "4096",
        "--mem-fraction-static", str(mem_fraction),
        "--page-size", "16",
        "--disable-cuda-graph",
        "--stream-interval", "100",
        "--log-level", "debug" if debug else "info",
    ]


def stream_output(proc: subprocess.Popen, name: str, log_file: str):
    """Stream stdout from a process to a log file."""
    with open(log_file, 'w') as f:
        for line in iter(proc.stdout.readline, ""):
            if line:
                f.write(line)
                f.flush()


def start_worker(
    name: str, args: list[str], env: dict[str, str] | None = None
) -> subprocess.Popen:
    """Start a worker process and stream its logs to a file."""
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=proc_env,
    )

    processes[name] = proc

    # Create log file in /tmp/
    log_file = f"/tmp/sglang_{name}.log"
    print(f"📝 Logging {name} output to {log_file}")

    # Start thread to stream output to file
    threading.Thread(
        target=stream_output, args=(proc, name, log_file), daemon=True
    ).start()

    return proc


def monitor_processes():
    """Monitor all processes and signal stop if any exits."""
    while not stop_event.is_set():
        for name, proc in list(processes.items()):
            ret = proc.poll()
            if ret is not None:
                print(f"\n💀 [{name}] exited with code {ret}")
                print(f"   See /tmp/sglang_{name}.log for details")
                stop_event.set()
                return
        time.sleep(0.5)


def wait_for_health(num_workers: int, timeout: float = 300.0):
    """Poll /health until we have the expected number of instances."""
    print(f"\n⏳ Waiting for {num_workers} workers to be ready...")
    start = time.time()

    while time.time() - start < timeout:
        if stop_event.is_set():
            print("\n❌ A process exited, aborting health check")
            return False

        try:
            resp = httpx.get(HEALTH_URL, timeout=2.0)
            data = resp.json()
            instances = data.get("instances", [])
            count = len(set([a["instance_id"] for a in instances]))

            if count >= num_workers:
                print(f"✅ All {num_workers} workers are ready!")
                return True
            else:
                print(f"   ... {count}/{num_workers} workers ready", end="\r")
        except Exception:
            print("   ... waiting for frontend to start", end="\r")

        time.sleep(1.0)

    print(f"\n❌ Timeout waiting for workers after {timeout}s")
    return False


def run_request(max_tokens: int = 50) -> bool:
    """Run a chat completion request. Returns True on success."""
    topics = [
        "Tell me a story about",
        "Explain how to",
        "Write a poem about",
    ]
    subjects = [
        "a brave knight",
        "cooking pasta",
        "the ocean",
    ]
    prompt = f"{random.choice(topics)} {random.choice(subjects)}"
    
    print(f"\n📝 Prompt: {prompt}")
    print(f"🎯 Max tokens: {max_tokens}")

    payload = {
        "model": "model",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    try:
        full_output = []
        with httpx.stream("POST", CHAT_URL, json=payload, timeout=120.0) as resp:
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_output.append(content)
                    except json.JSONDecodeError:
                        pass
        
        final_text = "".join(full_output)
        print(f"✅ Response ({len(final_text)} chars): {final_text[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False


def cleanup():
    """Terminate all running processes."""
    print("\n🧹 Cleaning up processes...")
    for name, proc in processes.items():
        if proc.poll() is None:
            proc.terminate()
    for name, proc in processes.items():
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("✅ Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Test mixed-TP disaggregation with Triton KV transfer"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to model",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["triton", "legacy"],
        default="legacy",
        help="KV transfer method to test",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=5,
        help="Number of test requests",
    )
    parser.add_argument(
        "--reverse-tp",
        action="store_true",
        help="Reverse TP: Prefill TP=1 (GPU 0) -> Decode TP=2 (GPU 1,2)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for workers",
    )
    parser.add_argument(
        "--decode-disagg",
        action="store_true",
        help="Enable decode disaggregation (short->long migration). Uses 8 GPUs.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max tokens per request (use >512 to trigger migration in decode-disagg mode)",
    )
    args = parser.parse_args()

    # Clean up any stale processes from previous runs
    cleanup_stale_processes()

    model_path = Path(args.model_path)

    # Decode disagg mode configuration (3 GPUs, no sharing)
    if args.decode_disagg:
        prefill_tp = 1
        decode_short_tp = 1
        decode_long_tp = 1
        prefill_gpus = "0"
        decode_short_gpus = "1"
        decode_long_gpus = "2"  # Each worker on separate GPU
        setup_desc = f"Prefill TP={prefill_tp} -> Decode Short TP={decode_short_tp} -> Decode Long TP={decode_long_tp} (separate GPUs)"
        num_workers = 3
    elif args.reverse_tp:
        prefill_tp = 1
        decode_tp = 2
        prefill_gpus = "0"
        decode_gpus = "1,2"
        setup_desc = "Prefill TP=1 (GPU 0) -> Decode TP=2 (GPU 1,2)"
        num_workers = 2
    else:
        prefill_tp = 2
        decode_tp = 1
        prefill_gpus = "0,1"
        decode_gpus = "2"
        setup_desc = "Prefill TP=2 (GPU 0,1) -> Decode TP=1 (GPU 2)"
        num_workers = 2

    print("=" * 60)
    print("Mixed-TP Disaggregation Test")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"KV Transfer Method: {args.method}")
    print(f"Setup: {setup_desc}")
    print(f"Decode Disagg: {args.decode_disagg}")
    if args.decode_disagg:
        print(f"Tier Boundaries: {DECODE_DISAGG_TIERS}")
    print(f"Debug logging: {args.debug}")
    print(f"Python: {PYTHON_EXE}")
    print("=" * 60)

    # Memory fractions depend on whether workers share GPUs
    if args.decode_disagg:
        # Each worker on separate GPU - can use more memory
        prefill_mem_frac = 0.7
        decode_mem_frac = 0.7
    else:
        prefill_mem_frac = 0.4
        decode_mem_frac = 0.4

    common_args = get_common_args(model_path, debug=args.debug, mem_fraction=prefill_mem_frac)
    decode_common_args = get_common_args(model_path, debug=args.debug, mem_fraction=decode_mem_frac)

    # UCX environment to force network transfers (skip shared memory)
    ucx_env = {"UCX_TLS": "^shm"}

    # Port configuration (spread apart to avoid conflicts)
    prefill_port = 10000
    decode_port = 11500
    decode_short_port = 11500
    decode_long_port = 13000
    prefill_bootstrap_port = 12345
    decode_bootstrap_port = 12346
    decode_short_bootstrap_port = 12346
    decode_long_bootstrap_port = 12347
    prefill_nccl_port = 29500
    decode_nccl_port = 29600
    decode_short_nccl_port = 29600
    decode_long_nccl_port = 29700

    try:
        # Start prefill worker
        print(f"\n🚀 Starting prefill worker (TP={prefill_tp}, GPU {prefill_gpus})...")
        prefill_args = [
            PYTHON_EXE, "-m", "dynamo.sglang",
            *common_args,
            "--tp", str(prefill_tp),
            "--load-balance-method", "round_robin",
            "--disaggregation-mode", "prefill",
            "--disaggregation-bootstrap-port", str(prefill_bootstrap_port),
            "--disaggregation-transfer-backend", "nixl",
            "--kv-transfer-method", args.method,
            "--pinned-buffer-max-gb", "2.0",
            "--host", "0.0.0.0",
            "--port", str(prefill_port),
            "--nccl-port", str(prefill_nccl_port),
        ]
        start_worker("prefill", prefill_args, env={
            "CUDA_VISIBLE_DEVICES": prefill_gpus,
            "DYN_SYSTEM_PORT": "8081",
        })

        if args.decode_disagg:
            # Decode disagg mode: start two decode workers with tier boundaries
            print(f"🚀 Starting decode_short worker (TP={decode_short_tp}, GPU {decode_short_gpus}, tier <= {DECODE_DISAGG_TIERS[0]})...")
            decode_short_args = [
                PYTHON_EXE, "-m", "dynamo.sglang",
                *decode_common_args,
                "--tp", str(decode_short_tp),
                "--prefill-round-robin-balance",
                "--disaggregation-mode", "decode",
                "--disaggregation-bootstrap-port", str(decode_short_bootstrap_port),
                "--disaggregation-transfer-backend", "nixl",
                "--kv-transfer-method", args.method,
                "--pinned-buffer-max-gb", "2.0",
                "--host", "0.0.0.0",
                "--port", str(decode_short_port),
                "--nccl-port", str(decode_short_nccl_port),
                "--this-seqlen", str(DECODE_DISAGG_TIERS[0]),
            ]
            start_worker("decode_short", decode_short_args, env={
                "CUDA_VISIBLE_DEVICES": decode_short_gpus,
                "DYN_SYSTEM_PORT": "8082",
                **ucx_env,
            })

            print(f"🚀 Starting decode_long worker (TP={decode_long_tp}, GPU {decode_long_gpus}, tier <= {DECODE_DISAGG_TIERS[1]})...")
            decode_long_args = [
                PYTHON_EXE, "-m", "dynamo.sglang",
                *decode_common_args,
                "--tp", str(decode_long_tp),
                "--prefill-round-robin-balance",
                "--disaggregation-mode", "decode",
                "--disaggregation-bootstrap-port", str(decode_long_bootstrap_port),
                "--disaggregation-transfer-backend", "nixl",
                "--kv-transfer-method", args.method,
                "--pinned-buffer-max-gb", "2.0",
                "--host", "0.0.0.0",
                "--port", str(decode_long_port),
                "--nccl-port", str(decode_long_nccl_port),
                "--this-seqlen", str(DECODE_DISAGG_TIERS[1]),
            ]
            start_worker("decode_long", decode_long_args, env={
                "CUDA_VISIBLE_DEVICES": decode_long_gpus,
                "DYN_SYSTEM_PORT": "8083",
                **ucx_env,
            })
        else:
            # Standard mode: single decode worker
            print(f"🚀 Starting decode worker (TP={decode_tp}, GPU {decode_gpus})...")
            decode_args = [
                PYTHON_EXE, "-m", "dynamo.sglang",
                *decode_common_args,
                "--tp", str(decode_tp),
                "--prefill-round-robin-balance",
                "--disaggregation-mode", "decode",
                "--disaggregation-bootstrap-port", str(decode_bootstrap_port),
                "--disaggregation-transfer-backend", "nixl",
                "--kv-transfer-method", args.method,
                "--pinned-buffer-max-gb", "2.0",
                "--host", "0.0.0.0",
                "--port", str(decode_port),
                "--nccl-port", str(decode_nccl_port),
            ]
            start_worker("decode", decode_args, env={
                "CUDA_VISIBLE_DEVICES": decode_gpus,
                "DYN_SYSTEM_PORT": "8082",
            })

        # Start frontend
        print("🚀 Starting frontend...")
        frontend_args = [
            PYTHON_EXE, "-m", "dynamo.frontend",
            "--http-port", "8080",
            "--namespace", "dynamo",
        ]
        if args.decode_disagg:
            frontend_args.extend(["--enable-decode-disagg", "--enforce-disagg"])
        start_worker("frontend", frontend_args)

        # Start process monitor
        monitor_thread = threading.Thread(target=monitor_processes, daemon=True)
        monitor_thread.start()

        # Wait for health
        if not wait_for_health(num_workers):
            print("Failed to start workers")
            return 1

        print("\n" + "=" * 60)
        print(f"Running {args.num_requests} test requests (max_tokens={args.max_tokens})...")
        if args.decode_disagg:
            print(f"Note: Use --max-tokens > {DECODE_DISAGG_TIERS[0]} to trigger migration")
        print("=" * 60)

        # Run test requests
        successes = 0
        for i in range(args.num_requests):
            print(f"\n--- Request {i+1}/{args.num_requests} ---")
            if run_request(max_tokens=args.max_tokens):
                successes += 1
            time.sleep(1)

            if stop_event.is_set():
                print("\n⚠️ Process failure detected, stopping test")
                break

        print("\n" + "=" * 60)
        print(f"RESULTS: {successes}/{args.num_requests} requests succeeded")
        print("=" * 60)

        return 0 if successes == args.num_requests else 1

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        return 1
    finally:
        stop_event.set()
        cleanup()


if __name__ == "__main__":
    sys.exit(main())
