#!/usr/bin/env python3
"""
Test script to reproduce EAGLE DP bug in disaggregation mode.
Uses nixl transfer backend.
"""

import subprocess
import time
import requests
import sys
import os
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL_PATH = os.path.expanduser("~/proj/models/dsv2-lite-fp8")
PREFILL_PORT = 30000
DECODE_PORT = 35000
LB_PORT = 8080

processes = []

def cleanup():
    print("\n[CLEANUP] Killing all processes...")
    for p in processes:
        try:
            p.terminate()
            p.wait(timeout=5)
        except:
            try:
                p.kill()
            except:
                pass
    os.system("pkill -9 -f 'sglang.launch_server' 2>/dev/null")
    os.system("pkill -9 -f 'sglang_router' 2>/dev/null")
    time.sleep(2)

def signal_handler(sig, frame):
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def wait_for_server(url, name, timeout=300):
    print(f"[{name}] Waiting for {url}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                print(f"[{name}] Ready!")
                return True
        except:
            pass
        time.sleep(2)
    print(f"[{name}] Failed to start within {timeout}s")
    return False

def send_request(session, port, req_id, prompt_tokens=50, max_tokens=100):
    prompt = "Write a story. " + " ".join([f"word{i}" for i in range(prompt_tokens)])
    payload = {"model": "dsv2", "prompt": prompt, "max_tokens": max_tokens}
    try:
        resp = session.post(f"http://127.0.0.1:{port}/v1/completions", json=payload, timeout=120)
        result = resp.json()
        if "error" in result:
            return False, f"Error: {result.get('error', result)}"
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("EAGLE DP Disaggregation Bug Reproduction")
    print("=" * 60)
    
    cleanup()
    
    # Start prefill worker on GPU 0
    print("\n[1] Starting PREFILL worker...")
    prefill_cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", MODEL_PATH,
        "--port", str(PREFILL_PORT),
        "--host", "127.0.0.1",
        "--disaggregation-mode", "prefill",
        "--disaggregation-transfer-backend", "nixl",
        "--mem-fraction-static", "0.25",
        "--tp-size", "1",
        "--disable-cuda-graph",
        "--max-running-requests", "50",
    ]
    
    prefill_env = os.environ.copy()
    prefill_env["CUDA_VISIBLE_DEVICES"] = "0"
    
    prefill_proc = subprocess.Popen(
        prefill_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=prefill_env,
        text=True,
        bufsize=1,
    )
    processes.append(prefill_proc)
    
    if not wait_for_server(f"http://127.0.0.1:{PREFILL_PORT}/health", "PREFILL", timeout=180):
        # Print last output
        print("[PREFILL] Failed. Output:")
        try:
            out, _ = prefill_proc.communicate(timeout=5)
            print(out[-5000:] if len(out) > 5000 else out)
        except:
            pass
        cleanup()
        return 1
    
    # Start decode worker on GPU 0,1 with DP attention + EAGLE
    print("\n[2] Starting DECODE worker with DP + EAGLE...")
    decode_cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", MODEL_PATH,
        "--port", str(DECODE_PORT),
        "--host", "127.0.0.1",
        "--disaggregation-mode", "decode",
        "--disaggregation-transfer-backend", "nixl",
        "--mem-fraction-static", "0.25",
        "--tp-size", "2",
        "--dp-size", "2",
        "--enable-dp-attention",
        "--base-gpu-id", "0",
        "--max-running-requests", "25",
        "--speculative-algorithm", "EAGLE",
        "--speculative-num-steps", "2",
        "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", "3",
    ]
    
    decode_env = os.environ.copy()
    decode_env["CUDA_VISIBLE_DEVICES"] = "0,1"
    decode_env["EAGLE_DEBUG"] = "1"
    
    decode_proc = subprocess.Popen(
        decode_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=decode_env,
        text=True,
        bufsize=1,
    )
    processes.append(decode_proc)
    
    if not wait_for_server(f"http://127.0.0.1:{DECODE_PORT}/health", "DECODE", timeout=300):
        print("[DECODE] Failed. Output:")
        try:
            out, _ = decode_proc.communicate(timeout=5)
            print(out[-5000:] if len(out) > 5000 else out)
        except:
            pass
        cleanup()
        return 1
    
    # Start load balancer
    print("\n[3] Starting load balancer...")
    lb_cmd = [
        sys.executable, "-m", "sglang_router.launch_router",
        "--pd-disaggregation",
        "--mini-lb",
        "--prefill", f"http://127.0.0.1:{PREFILL_PORT}",
        "--decode", f"http://127.0.0.1:{DECODE_PORT}",
        "--host", "127.0.0.1",
        "--port", str(LB_PORT),
    ]
    
    lb_proc = subprocess.Popen(
        lb_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    processes.append(lb_proc)
    
    if not wait_for_server(f"http://127.0.0.1:{LB_PORT}/health", "LB", timeout=60):
        print("[LB] Failed. Output:")
        try:
            out, _ = lb_proc.communicate(timeout=5)
            print(out[-2000:] if len(out) > 2000 else out)
        except:
            pass
        cleanup()
        return 1
    
    print("\n" + "=" * 60)
    print("All servers ready! Running load test...")
    print("=" * 60)
    
    # Run load test
    session = requests.Session()
    total_success = 0
    total_failed = 0
    
    for round_num in range(20):
        with ThreadPoolExecutor(max_workers=25) as executor:
            futures = []
            for i in range(25):
                prompt_tokens = 20 + (i % 80)
                futures.append(executor.submit(send_request, session, LB_PORT, round_num * 25 + i, prompt_tokens))
            
            successes = failures = 0
            for future in as_completed(futures):
                success, error = future.result()
                if success:
                    successes += 1
                else:
                    failures += 1
                    if error:
                        print(f"  Error: {error[:100]}")
            
            total_success += successes
            total_failed += failures
            print(f"Round {round_num + 1}/20: {successes}/25 succeeded")
            
            if failures > 0:
                print("\n!!! FAILURES DETECTED - Checking for CUDA errors !!!")
                break
        
        time.sleep(0.5)
    
    print(f"\nTotal: {total_success} succeeded, {total_failed} failed")
    
    # Check for CUDA errors in logs
    print("\n[4] Checking process outputs for CUDA errors...")
    for name, proc in [("PREFILL", prefill_proc), ("DECODE", decode_proc)]:
        if proc.poll() is not None:
            print(f"[{name}] Process exited with code {proc.returncode}")
    
    time.sleep(5)
    cleanup()
    
    return 0 if total_failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
