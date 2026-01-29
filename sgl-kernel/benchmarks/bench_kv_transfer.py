"""
Benchmark for KV transfer Triton kernels.

Measures bandwidth achieved when gathering scattered KV cache data
from GPU and writing directly to pinned CPU memory.
"""

import argparse
import sys
from pathlib import Path

import torch
import triton

# Add the triton_ops path
triton_ops_path = Path(__file__).parent.parent.parent / "python" / "sglang" / "srt" / "layers" / "attention" / "triton_ops"
sys.path.insert(0, str(triton_ops_path))

from kv_transfer import gather_kv, gather_kv_to_pinned, gather_kv_with_staging, gather_kv_chunked, scatter_kv_to_gpu


def benchmark_gather_kv(
    num_layers: int,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    total_slots: int,
    head_start: int,
    num_heads_to_gather: int,
    pattern: str,
    dtype: torch.dtype,
    warmup: int = 10,
    rep: int = 100,
) -> dict:
    """
    Benchmark the gather_kv_to_pinned kernel.

    Returns dict with timing and bandwidth metrics.
    """
    # Create KV buffers
    k_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    # Create slot indices based on pattern
    if pattern == "contiguous":
        start = total_slots // 4
        slot_indices = torch.arange(start, start + num_tokens, device="cuda", dtype=torch.int32)
    elif pattern == "random":
        slot_indices = torch.randperm(total_slots, device="cuda")[:num_tokens].to(torch.int32)
    elif pattern == "strided":
        stride = max(1, total_slots // num_tokens)
        slot_indices = torch.arange(0, num_tokens * stride, stride, device="cuda", dtype=torch.int32)
        slot_indices = slot_indices[:num_tokens]  # Ensure correct count
    elif pattern == "reverse":
        slot_indices = torch.arange(num_tokens - 1, -1, -1, device="cuda", dtype=torch.int32) * (total_slots // num_tokens)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    # Allocate pinned output
    output_size = num_layers * 2 * num_tokens * num_heads_to_gather * head_dim
    pinned_output = torch.empty(output_size, dtype=dtype, device="cpu", pin_memory=True)

    # Calculate data size
    bytes_per_element = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    total_bytes = num_layers * 2 * num_tokens * num_heads_to_gather * head_dim * bytes_per_element

    # Warmup
    for _ in range(warmup):
        gather_kv_to_pinned(
            k_buffers=k_buffers,
            v_buffers=v_buffers,
            slot_indices=slot_indices,
            pinned_output=pinned_output,
            head_start=head_start,
            num_heads_to_gather=num_heads_to_gather,
        )

    # Benchmark using triton's benchmarking utility
    torch.cuda.synchronize()

    # Manual timing for more control
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    for i in range(rep):
        start_events[i].record()
        gather_kv_to_pinned(
            k_buffers=k_buffers,
            v_buffers=v_buffers,
            slot_indices=slot_indices,
            pinned_output=pinned_output,
            head_start=head_start,
            num_heads_to_gather=num_heads_to_gather,
        )
        end_events[i].record()

    torch.cuda.synchronize()

    times_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(rep)]
    avg_time_ms = sum(times_ms) / len(times_ms)
    min_time_ms = min(times_ms)
    max_time_ms = max(times_ms)

    # Calculate bandwidth (GB/s)
    avg_bandwidth_gbs = (total_bytes / 1e9) / (avg_time_ms / 1000)
    peak_bandwidth_gbs = (total_bytes / 1e9) / (min_time_ms / 1000)

    # Clean up
    del k_buffers, v_buffers, pinned_output
    torch.cuda.empty_cache()

    return {
        "num_layers": num_layers,
        "num_tokens": num_tokens,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "total_slots": total_slots,
        "num_heads_to_gather": num_heads_to_gather,
        "pattern": pattern,
        "dtype": str(dtype),
        "total_bytes_mb": total_bytes / 1e6,
        "avg_time_ms": avg_time_ms,
        "min_time_ms": min_time_ms,
        "max_time_ms": max_time_ms,
        "avg_bandwidth_gbs": avg_bandwidth_gbs,
        "peak_bandwidth_gbs": peak_bandwidth_gbs,
    }


def benchmark_gather_with_staging(
    num_layers: int,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    total_slots: int,
    head_start: int,
    num_heads_to_gather: int,
    pattern: str,
    dtype: torch.dtype,
    warmup: int = 10,
    rep: int = 100,
) -> dict:
    """
    Benchmark the gather_kv_with_staging kernel (GPU staging buffer approach).
    """
    k_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    if pattern == "contiguous":
        start = total_slots // 4
        slot_indices = torch.arange(start, start + num_tokens, device="cuda", dtype=torch.int32)
    elif pattern == "random":
        slot_indices = torch.randperm(total_slots, device="cuda")[:num_tokens].to(torch.int32)
    elif pattern == "strided":
        stride = max(1, total_slots // num_tokens)
        slot_indices = torch.arange(0, num_tokens * stride, stride, device="cuda", dtype=torch.int32)[:num_tokens]
    else:
        slot_indices = torch.randperm(total_slots, device="cuda")[:num_tokens].to(torch.int32)

    # Allocate buffers
    bytes_per_element = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    total_bytes = num_layers * 2 * num_tokens * num_heads_to_gather * head_dim * bytes_per_element

    staging_size = num_layers * 2 * num_tokens * num_heads_to_gather * head_dim
    staging_buffer = torch.empty(staging_size, dtype=dtype, device="cuda")
    pinned_output = torch.empty(staging_size, dtype=dtype, device="cpu", pin_memory=True)

    # Warmup
    for _ in range(warmup):
        gather_kv_with_staging(
            k_buffers=k_buffers,
            v_buffers=v_buffers,
            slot_indices=slot_indices,
            pinned_output=pinned_output,
            head_start=head_start,
            num_heads_to_gather=num_heads_to_gather,
            staging_buffer=staging_buffer,
        )

    # Benchmark
    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    for i in range(rep):
        start_events[i].record()
        gather_kv_with_staging(
            k_buffers=k_buffers,
            v_buffers=v_buffers,
            slot_indices=slot_indices,
            pinned_output=pinned_output,
            head_start=head_start,
            num_heads_to_gather=num_heads_to_gather,
            staging_buffer=staging_buffer,
        )
        end_events[i].record()

    torch.cuda.synchronize()

    times_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(rep)]
    avg_time_ms = sum(times_ms) / len(times_ms)
    min_time_ms = min(times_ms)

    avg_bandwidth_gbs = (total_bytes / 1e9) / (avg_time_ms / 1000)
    peak_bandwidth_gbs = (total_bytes / 1e9) / (min_time_ms / 1000)

    del k_buffers, v_buffers, staging_buffer, pinned_output
    torch.cuda.empty_cache()

    return {
        "num_layers": num_layers,
        "num_tokens": num_tokens,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "pattern": pattern,
        "total_bytes_mb": total_bytes / 1e6,
        "avg_time_ms": avg_time_ms,
        "min_time_ms": min_time_ms,
        "avg_bandwidth_gbs": avg_bandwidth_gbs,
        "peak_bandwidth_gbs": peak_bandwidth_gbs,
        "staging_buffer_mb": staging_size * bytes_per_element / 1e6,
    }


def benchmark_gather_chunked(
    num_layers: int,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    total_slots: int,
    head_start: int,
    num_heads_to_gather: int,
    pattern: str,
    dtype: torch.dtype,
    staging_buffer_size_mb: float = 256.0,
    warmup: int = 10,
    rep: int = 100,
) -> dict:
    """
    Benchmark the gather_kv_chunked kernel (fixed-size staging buffer approach).
    """
    k_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    if pattern == "contiguous":
        start = total_slots // 4
        slot_indices = torch.arange(start, start + num_tokens, device="cuda", dtype=torch.int32)
    elif pattern == "random":
        slot_indices = torch.randperm(total_slots, device="cuda")[:num_tokens].to(torch.int32)
    elif pattern == "strided":
        stride = max(1, total_slots // num_tokens)
        slot_indices = torch.arange(0, num_tokens * stride, stride, device="cuda", dtype=torch.int32)[:num_tokens]
    else:
        slot_indices = torch.randperm(total_slots, device="cuda")[:num_tokens].to(torch.int32)

    # Allocate buffers
    bytes_per_element = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    total_bytes = num_layers * 2 * num_tokens * num_heads_to_gather * head_dim * bytes_per_element

    output_size = num_layers * 2 * num_tokens * num_heads_to_gather * head_dim
    pinned_output = torch.empty(output_size, dtype=dtype, device="cpu", pin_memory=True)

    # Calculate actual staging buffer size used
    single_kv_elements = num_tokens * num_heads_to_gather * head_dim
    staging_buffer_bytes = int(staging_buffer_size_mb * 1e6)
    staging_buffer_elements = staging_buffer_bytes // bytes_per_element
    kvs_per_chunk = max(1, staging_buffer_elements // single_kv_elements)
    actual_staging_elements = kvs_per_chunk * single_kv_elements
    staging_buffer = torch.empty(actual_staging_elements, dtype=dtype, device="cuda")

    # Warmup
    for _ in range(warmup):
        gather_kv_chunked(
            k_buffers=k_buffers,
            v_buffers=v_buffers,
            slot_indices=slot_indices,
            pinned_output=pinned_output,
            head_start=head_start,
            num_heads_to_gather=num_heads_to_gather,
            staging_buffer_size_mb=staging_buffer_size_mb,
            staging_buffer=staging_buffer,
        )

    # Benchmark
    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    for i in range(rep):
        start_events[i].record()
        gather_kv_chunked(
            k_buffers=k_buffers,
            v_buffers=v_buffers,
            slot_indices=slot_indices,
            pinned_output=pinned_output,
            head_start=head_start,
            num_heads_to_gather=num_heads_to_gather,
            staging_buffer_size_mb=staging_buffer_size_mb,
            staging_buffer=staging_buffer,
        )
        end_events[i].record()

    torch.cuda.synchronize()

    times_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(rep)]
    avg_time_ms = sum(times_ms) / len(times_ms)
    min_time_ms = min(times_ms)

    avg_bandwidth_gbs = (total_bytes / 1e9) / (avg_time_ms / 1000)
    peak_bandwidth_gbs = (total_bytes / 1e9) / (min_time_ms / 1000)

    del k_buffers, v_buffers, staging_buffer, pinned_output
    torch.cuda.empty_cache()

    return {
        "num_layers": num_layers,
        "num_tokens": num_tokens,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "pattern": pattern,
        "total_bytes_mb": total_bytes / 1e6,
        "avg_time_ms": avg_time_ms,
        "min_time_ms": min_time_ms,
        "avg_bandwidth_gbs": avg_bandwidth_gbs,
        "peak_bandwidth_gbs": peak_bandwidth_gbs,
        "staging_buffer_mb": actual_staging_elements * bytes_per_element / 1e6,
        "kvs_per_chunk": kvs_per_chunk,
    }


def benchmark_pytorch_baseline(
    num_layers: int,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    total_slots: int,
    pattern: str,
    dtype: torch.dtype,
    warmup: int = 10,
    rep: int = 100,
) -> dict:
    """
    Benchmark baseline PyTorch gather + copy for comparison.
    """
    k_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    if pattern == "contiguous":
        start = total_slots // 4
        slot_indices = torch.arange(start, start + num_tokens, device="cuda", dtype=torch.int64)
    elif pattern == "random":
        slot_indices = torch.randperm(total_slots, device="cuda")[:num_tokens]
    elif pattern == "strided":
        stride = max(1, total_slots // num_tokens)
        slot_indices = torch.arange(0, num_tokens * stride, stride, device="cuda", dtype=torch.int64)[:num_tokens]
    else:
        slot_indices = torch.randperm(total_slots, device="cuda")[:num_tokens]

    pinned_output = torch.empty(
        (num_layers, 2, num_tokens, num_heads, head_dim),
        dtype=dtype, device="cpu", pin_memory=True
    )

    bytes_per_element = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    total_bytes = num_layers * 2 * num_tokens * num_heads * head_dim * bytes_per_element

    def pytorch_gather():
        for layer_idx in range(num_layers):
            # Gather on GPU then copy to pinned CPU
            k_gathered = k_buffers[layer_idx][slot_indices]
            v_gathered = v_buffers[layer_idx][slot_indices]
            pinned_output[layer_idx, 0].copy_(k_gathered, non_blocking=True)
            pinned_output[layer_idx, 1].copy_(v_gathered, non_blocking=True)
        torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        pytorch_gather()

    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    for i in range(rep):
        start_events[i].record()
        pytorch_gather()
        end_events[i].record()

    torch.cuda.synchronize()

    times_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(rep)]
    avg_time_ms = sum(times_ms) / len(times_ms)
    min_time_ms = min(times_ms)

    avg_bandwidth_gbs = (total_bytes / 1e9) / (avg_time_ms / 1000)
    peak_bandwidth_gbs = (total_bytes / 1e9) / (min_time_ms / 1000)

    # Clean up
    del k_buffers, v_buffers, pinned_output
    torch.cuda.empty_cache()

    return {
        "method": "pytorch_baseline",
        "pattern": pattern,
        "total_bytes_mb": total_bytes / 1e6,
        "avg_time_ms": avg_time_ms,
        "min_time_ms": min_time_ms,
        "avg_bandwidth_gbs": avg_bandwidth_gbs,
        "peak_bandwidth_gbs": peak_bandwidth_gbs,
    }


def print_results_table(results: list[dict], title: str):
    """Print results in a formatted table."""
    print(f"\n{'=' * 80}")
    print(f" {title}")
    print('=' * 80)

    # Header
    print(f"{'Pattern':<12} {'Tokens':<8} {'Layers':<7} {'Size(MB)':<10} "
          f"{'Time(ms)':<10} {'BW(GB/s)':<10} {'Peak BW':<10}")
    print('-' * 80)

    for r in results:
        print(f"{r['pattern']:<12} {r.get('num_tokens', '-'):<8} {r.get('num_layers', '-'):<7} "
              f"{r['total_bytes_mb']:<10.2f} {r['avg_time_ms']:<10.3f} "
              f"{r['avg_bandwidth_gbs']:<10.2f} {r['peak_bandwidth_gbs']:<10.2f}")


def benchmark_cuda_memcpy(
    size_mb: float,
    dtype: torch.dtype,
    warmup: int = 10,
    rep: int = 100,
) -> dict:
    """
    Benchmark raw CUDA memcpy from GPU to pinned CPU.

    This represents the theoretical maximum PCIe bandwidth achievable.
    """
    bytes_per_element = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    num_elements = int(size_mb * 1e6 / bytes_per_element)
    total_bytes = num_elements * bytes_per_element

    # Contiguous GPU tensor
    gpu_tensor = torch.randn(num_elements, dtype=dtype, device="cuda")

    # Pinned CPU tensor
    cpu_tensor = torch.empty(num_elements, dtype=dtype, device="cpu", pin_memory=True)

    # Warmup
    for _ in range(warmup):
        cpu_tensor.copy_(gpu_tensor, non_blocking=False)
        torch.cuda.synchronize()

    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    for i in range(rep):
        start_events[i].record()
        cpu_tensor.copy_(gpu_tensor, non_blocking=False)
        end_events[i].record()

    torch.cuda.synchronize()

    times_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(rep)]
    avg_time_ms = sum(times_ms) / len(times_ms)
    min_time_ms = min(times_ms)

    avg_bandwidth_gbs = (total_bytes / 1e9) / (avg_time_ms / 1000)
    peak_bandwidth_gbs = (total_bytes / 1e9) / (min_time_ms / 1000)

    del gpu_tensor, cpu_tensor
    torch.cuda.empty_cache()

    return {
        "size_mb": size_mb,
        "avg_time_ms": avg_time_ms,
        "min_time_ms": min_time_ms,
        "avg_bandwidth_gbs": avg_bandwidth_gbs,
        "peak_bandwidth_gbs": peak_bandwidth_gbs,
    }


def benchmark_memory_usage(
    num_layers: int,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    total_slots: int,
    dtype: torch.dtype,
) -> dict:
    """
    Compare GPU memory usage between Triton (zero-copy) and PyTorch (staging) approaches.
    """
    bytes_per_element = 2 if dtype in (torch.float16, torch.bfloat16) else 4

    # KV cache size (same for both approaches)
    kv_cache_bytes = num_layers * 2 * total_slots * num_heads * head_dim * bytes_per_element

    # Transfer size
    transfer_bytes = num_layers * 2 * num_tokens * num_heads * head_dim * bytes_per_element

    # Triton approach: zero GPU staging, only pinned CPU buffer
    triton_gpu_staging = 0
    triton_cpu_pinned = transfer_bytes

    # PyTorch approach: needs GPU staging buffer for gathered tensors
    pytorch_gpu_staging = transfer_bytes  # Intermediate GPU tensor
    pytorch_cpu_pinned = transfer_bytes

    return {
        "kv_cache_mb": kv_cache_bytes / 1e6,
        "transfer_mb": transfer_bytes / 1e6,
        "triton_gpu_staging_mb": triton_gpu_staging / 1e6,
        "triton_cpu_pinned_mb": triton_cpu_pinned / 1e6,
        "pytorch_gpu_staging_mb": pytorch_gpu_staging / 1e6,
        "pytorch_cpu_pinned_mb": pytorch_cpu_pinned / 1e6,
        "gpu_memory_saved_mb": (pytorch_gpu_staging - triton_gpu_staging) / 1e6,
    }


def benchmark_scatter_kv(
    num_layers: int,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    total_slots: int,
    head_start: int,
    num_heads_to_scatter: int,
    pattern: str,
    dtype: torch.dtype,
    warmup: int = 10,
    rep: int = 100,
) -> dict:
    """
    Benchmark the scatter_kv_to_gpu kernel (pinned CPU -> GPU).
    """
    # Create input in pinned memory
    bytes_per_element = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    total_bytes = num_layers * 2 * num_tokens * num_heads_to_scatter * head_dim * bytes_per_element

    input_size = num_layers * 2 * num_tokens * num_heads_to_scatter * head_dim
    pinned_input = torch.randn(input_size, dtype=dtype, device="cpu", pin_memory=True)

    # Create slot indices based on pattern
    if pattern == "contiguous":
        start = total_slots // 4
        slot_indices = torch.arange(start, start + num_tokens, device="cuda", dtype=torch.int32)
    elif pattern == "random":
        slot_indices = torch.randperm(total_slots, device="cuda")[:num_tokens].to(torch.int32)
    elif pattern == "strided":
        stride = max(1, total_slots // num_tokens)
        slot_indices = torch.arange(0, num_tokens * stride, stride, device="cuda", dtype=torch.int32)[:num_tokens]
    else:
        slot_indices = torch.randperm(total_slots, device="cuda")[:num_tokens].to(torch.int32)

    # Create destination KV buffers
    k_buffers = [
        torch.zeros(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers = [
        torch.zeros(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    # Warmup
    for _ in range(warmup):
        scatter_kv_to_gpu(
            pinned_input=pinned_input,
            k_buffers=k_buffers,
            v_buffers=v_buffers,
            slot_indices=slot_indices,
            head_start=head_start,
            num_heads_to_scatter=num_heads_to_scatter,
        )

    # Benchmark
    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    for i in range(rep):
        start_events[i].record()
        scatter_kv_to_gpu(
            pinned_input=pinned_input,
            k_buffers=k_buffers,
            v_buffers=v_buffers,
            slot_indices=slot_indices,
            head_start=head_start,
            num_heads_to_scatter=num_heads_to_scatter,
        )
        end_events[i].record()

    torch.cuda.synchronize()

    times_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(rep)]
    avg_time_ms = sum(times_ms) / len(times_ms)
    min_time_ms = min(times_ms)

    avg_bandwidth_gbs = (total_bytes / 1e9) / (avg_time_ms / 1000)
    peak_bandwidth_gbs = (total_bytes / 1e9) / (min_time_ms / 1000)

    del k_buffers, v_buffers, pinned_input
    torch.cuda.empty_cache()

    return {
        "num_layers": num_layers,
        "num_tokens": num_tokens,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "pattern": pattern,
        "total_bytes_mb": total_bytes / 1e6,
        "avg_time_ms": avg_time_ms,
        "min_time_ms": min_time_ms,
        "avg_bandwidth_gbs": avg_bandwidth_gbs,
        "peak_bandwidth_gbs": peak_bandwidth_gbs,
    }


def benchmark_large_pool_fragmented(
    staging_buffer_size_mb: float = 256.0,
    warmup: int = 5,
    rep: int = 20,
) -> dict:
    """
    Benchmark the realistic scenario: 92 layers, 32K tokens, 1024 bytes per K/V per layer,
    with a 40GB pool to simulate high fragmentation.

    This matches the user's scenario:
    - 1024 bytes per K/V per layer = 8 heads * 64 head_dim * 2 bytes (fp16)
    - 92 layers
    - 32K sequence length
    - 40GB pool (~20B slots at 2 bytes each, or ~2.5M slots with full KV)
    """
    dtype = torch.float16
    bytes_per_element = 2

    # 1024 bytes per K or V per layer = 512 elements (fp16)
    # 512 = num_heads * head_dim, so 8 heads * 64 head_dim = 512
    num_heads = 8
    head_dim = 64
    num_layers = 92
    num_tokens = 32768  # 32K

    # 40GB pool size - calculate how many slots we can fit
    # Each slot has K + V = 2 * num_heads * head_dim = 1024 elements = 2048 bytes per layer
    # But we have multiple layers, so per-slot is num_heads * head_dim elements
    pool_size_gb = 40
    pool_size_bytes = int(pool_size_gb * 1e9)

    # Each slot in the KV cache is [num_heads, head_dim] = 512 elements = 1024 bytes (fp16)
    bytes_per_slot_per_layer = num_heads * head_dim * bytes_per_element  # 1024 bytes
    # Total bytes per slot across all layers (for K only, or V only)
    # But our buffers are per-layer, each sized [total_slots, num_heads, head_dim]
    # So total_slots determines how big each layer's buffer is

    # To get 40GB total: num_layers * 2 (K+V) * total_slots * num_heads * head_dim * 2 bytes = 40GB
    # 92 * 2 * total_slots * 8 * 64 * 2 = 40e9
    # total_slots = 40e9 / (92 * 2 * 8 * 64 * 2) = 40e9 / 188416 = ~212,314 slots
    # But this doesn't give us fragmentation - we want MORE slots than we're using

    # Let's calculate differently: we want the pool to be 40GB
    # and we want to use only 32K slots from that pool (our sequence length)
    # So: total_slots = pool_size / (num_layers * 2 * bytes_per_slot)
    # Actually, we store K and V separately per layer, so:
    # Memory for one layer = total_slots * num_heads * head_dim * bytes_per_element
    # Memory for K = num_layers * total_slots * num_heads * head_dim * bytes_per_element
    # Memory for V = same
    # Total = 2 * num_layers * total_slots * num_heads * head_dim * bytes_per_element

    # total_slots = pool_size_bytes / (2 * num_layers * num_heads * head_dim * bytes_per_element)
    total_slots = pool_size_bytes // (2 * num_layers * num_heads * head_dim * bytes_per_element)
    print(f"\n  40GB pool configuration:")
    print(f"    total_slots = {total_slots:,} (pool can hold this many tokens)")
    print(f"    num_tokens = {num_tokens:,} (we're gathering this many)")
    print(f"    fragmentation ratio = {total_slots / num_tokens:.1f}x")

    # Calculate transfer size
    # Transfer = num_layers * 2 * num_tokens * num_heads * head_dim * bytes_per_element
    transfer_bytes = num_layers * 2 * num_tokens * num_heads * head_dim * bytes_per_element
    print(f"    transfer size = {transfer_bytes / 1e9:.2f} GB ({transfer_bytes / 1e6:.1f} MB)")

    # Create KV buffers (this will use ~40GB GPU memory)
    print(f"\n  Allocating {pool_size_gb}GB KV pool...")
    k_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    print(f"  Pool allocated. Free GPU memory: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

    # Create slot indices - random access pattern for maximum fragmentation
    slot_indices = torch.randperm(total_slots, device="cuda")[:num_tokens].to(torch.int32)

    # Allocate pinned output
    output_size = num_layers * 2 * num_tokens * num_heads * head_dim
    pinned_output = torch.empty(output_size, dtype=dtype, device="cpu", pin_memory=True)

    results = {}

    # Benchmark raw memcpy baseline
    print("\n  Benchmarking raw memcpy (contiguous baseline)...")
    contiguous_gpu = torch.randn(output_size, dtype=dtype, device="cuda")
    for _ in range(warmup):
        pinned_output.copy_(contiguous_gpu, non_blocking=False)
        torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        start_events[i].record()
        pinned_output.copy_(contiguous_gpu, non_blocking=False)
        end_events[i].record()
    torch.cuda.synchronize()

    times_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(rep)]
    memcpy_time_ms = sum(times_ms) / len(times_ms)
    memcpy_bw = (transfer_bytes / 1e9) / (memcpy_time_ms / 1000)
    results["memcpy"] = {"time_ms": memcpy_time_ms, "bandwidth_gbs": memcpy_bw}
    print(f"    Raw memcpy: {memcpy_time_ms:.1f} ms, {memcpy_bw:.2f} GB/s")
    del contiguous_gpu

    # Benchmark zero-copy (direct to pinned)
    print("\n  Benchmarking zero-copy (direct to pinned)...")
    for _ in range(warmup):
        gather_kv_to_pinned(
            k_buffers=k_buffers,
            v_buffers=v_buffers,
            slot_indices=slot_indices,
            pinned_output=pinned_output,
            head_start=0,
            num_heads_to_gather=num_heads,
        )

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        start_events[i].record()
        gather_kv_to_pinned(
            k_buffers=k_buffers,
            v_buffers=v_buffers,
            slot_indices=slot_indices,
            pinned_output=pinned_output,
            head_start=0,
            num_heads_to_gather=num_heads,
        )
        end_events[i].record()
    torch.cuda.synchronize()

    times_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(rep)]
    zerocopy_time_ms = sum(times_ms) / len(times_ms)
    zerocopy_bw = (transfer_bytes / 1e9) / (zerocopy_time_ms / 1000)
    results["zero_copy"] = {"time_ms": zerocopy_time_ms, "bandwidth_gbs": zerocopy_bw}
    print(f"    Zero-copy: {zerocopy_time_ms:.1f} ms, {zerocopy_bw:.2f} GB/s ({zerocopy_bw/memcpy_bw*100:.0f}% of memcpy)")

    # Benchmark chunked approach with different staging buffer sizes
    for staging_mb in [64, 128, 256, 512]:
        print(f"\n  Benchmarking chunked (staging={staging_mb}MB)...")

        # Pre-allocate staging buffer
        single_kv_elements = num_tokens * num_heads * head_dim
        staging_buffer_bytes = int(staging_mb * 1e6)
        staging_buffer_elements = staging_buffer_bytes // bytes_per_element
        kvs_per_chunk = max(1, staging_buffer_elements // single_kv_elements)
        actual_staging_elements = kvs_per_chunk * single_kv_elements
        staging_buffer = torch.empty(actual_staging_elements, dtype=dtype, device="cuda")
        actual_staging_mb = actual_staging_elements * bytes_per_element / 1e6

        print(f"      KVs per chunk: {kvs_per_chunk}, actual staging: {actual_staging_mb:.1f}MB")

        for _ in range(warmup):
            gather_kv_chunked(
                k_buffers=k_buffers,
                v_buffers=v_buffers,
                slot_indices=slot_indices,
                pinned_output=pinned_output,
                head_start=0,
                num_heads_to_gather=num_heads,
                staging_buffer_size_mb=staging_mb,
                staging_buffer=staging_buffer,
            )

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        for i in range(rep):
            start_events[i].record()
            gather_kv_chunked(
                k_buffers=k_buffers,
                v_buffers=v_buffers,
                slot_indices=slot_indices,
                pinned_output=pinned_output,
                head_start=0,
                num_heads_to_gather=num_heads,
                staging_buffer_size_mb=staging_mb,
                staging_buffer=staging_buffer,
            )
            end_events[i].record()
        torch.cuda.synchronize()

        times_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(rep)]
        chunked_time_ms = sum(times_ms) / len(times_ms)
        chunked_bw = (transfer_bytes / 1e9) / (chunked_time_ms / 1000)
        results[f"chunked_{staging_mb}mb"] = {
            "time_ms": chunked_time_ms,
            "bandwidth_gbs": chunked_bw,
            "staging_mb": actual_staging_mb,
            "kvs_per_chunk": kvs_per_chunk,
        }
        print(f"    Chunked ({staging_mb}MB): {chunked_time_ms:.1f} ms, {chunked_bw:.2f} GB/s ({chunked_bw/memcpy_bw*100:.0f}% of memcpy)")

        del staging_buffer

    # Benchmark scatter (host -> device)
    print("\n  Benchmarking scatter (pinned CPU -> GPU)...")

    # Fill pinned buffer with data to scatter
    pinned_input = torch.randn(output_size, dtype=dtype, device="cpu", pin_memory=True)

    # Create fresh destination buffers
    k_buffers_dst = [
        torch.zeros(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers_dst = [
        torch.zeros(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    for _ in range(warmup):
        scatter_kv_to_gpu(
            pinned_input=pinned_input,
            k_buffers=k_buffers_dst,
            v_buffers=v_buffers_dst,
            slot_indices=slot_indices,
            head_start=0,
            num_heads_to_scatter=num_heads,
        )

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        start_events[i].record()
        scatter_kv_to_gpu(
            pinned_input=pinned_input,
            k_buffers=k_buffers_dst,
            v_buffers=v_buffers_dst,
            slot_indices=slot_indices,
            head_start=0,
            num_heads_to_scatter=num_heads,
        )
        end_events[i].record()
    torch.cuda.synchronize()

    times_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(rep)]
    scatter_time_ms = sum(times_ms) / len(times_ms)
    scatter_bw = (transfer_bytes / 1e9) / (scatter_time_ms / 1000)
    results["scatter"] = {"time_ms": scatter_time_ms, "bandwidth_gbs": scatter_bw}
    print(f"    Scatter: {scatter_time_ms:.1f} ms, {scatter_bw:.2f} GB/s ({scatter_bw/memcpy_bw*100:.0f}% of memcpy)")

    del k_buffers_dst, v_buffers_dst, pinned_input

    # Clean up
    del k_buffers, v_buffers, pinned_output, slot_indices
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark KV transfer kernel")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=100, help="Benchmark repetitions")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark with fewer configs")
    parser.add_argument("--large-pool", action="store_true", help="Run 40GB pool fragmentation benchmark")
    parser.add_argument("--large-pool-only", action="store_true", help="Only run 40GB pool benchmark")
    args = parser.parse_args()

    print("=" * 80)
    print(" KV Transfer Kernel Benchmark")
    print(" GPU -> Pinned CPU via Triton kernel")
    print("=" * 80)

    # Get GPU info
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print(f"\nGPU: {props.name}")
    print(f"PCIe theoretical max: ~32 GB/s (Gen4 x16) or ~64 GB/s (Gen5 x16)")

    # Run large pool benchmark if requested
    if args.large_pool_only or args.large_pool:
        print("\n" + "=" * 80)
        print(" 40GB Pool Fragmentation Benchmark")
        print(" 92 layers, 32K tokens, high fragmentation scenario")
        print("=" * 80)
        results = benchmark_large_pool_fragmented(warmup=5, rep=20)

        print("\n" + "=" * 80)
        print(" Summary: 40GB Pool Fragmentation Test")
        print("=" * 80)
        memcpy_bw = results["memcpy"]["bandwidth_gbs"]
        print(f"\n {'Method':<25} {'Time (ms)':<12} {'BW (GB/s)':<12} {'% of memcpy':<12}")
        print("-" * 60)
        print(f" {'Raw memcpy':<25} {results['memcpy']['time_ms']:<12.1f} {results['memcpy']['bandwidth_gbs']:<12.2f} {'100%':<12}")
        print(f"\n Device -> Host (Gather):")
        print(f" {'Zero-copy':<25} {results['zero_copy']['time_ms']:<12.1f} {results['zero_copy']['bandwidth_gbs']:<12.2f} {results['zero_copy']['bandwidth_gbs']/memcpy_bw*100:.0f}%")
        for key in results:
            if key.startswith("chunked_"):
                r = results[key]
                name = f"Chunked ({r['staging_mb']:.0f}MB staging)"
                print(f" {name:<25} {r['time_ms']:<12.1f} {r['bandwidth_gbs']:<12.2f} {r['bandwidth_gbs']/memcpy_bw*100:.0f}%")
        if "scatter" in results:
            print(f"\n Host -> Device (Scatter):")
            print(f" {'Scatter (direct)':<25} {results['scatter']['time_ms']:<12.1f} {results['scatter']['bandwidth_gbs']:<12.2f} {results['scatter']['bandwidth_gbs']/memcpy_bw*100:.0f}%")

        if args.large_pool_only:
            return

    dtype = torch.float16
    patterns = ["contiguous", "random", "strided", "reverse"]

    if args.quick:
        configs = [
            # (num_layers, num_tokens, num_heads, head_dim, total_slots, num_heads_to_gather)
            (4, 1024, 8, 128, 100_000, 8),
            (32, 2048, 8, 128, 200_000, 8),
        ]
    else:
        configs = [
            # Small transfers
            (4, 128, 8, 128, 50_000, 8),
            (4, 512, 8, 128, 100_000, 8),

            # Medium transfers (typical use case)
            (4, 1024, 8, 128, 100_000, 8),
            (4, 2048, 8, 128, 200_000, 8),
            (4, 4096, 8, 128, 200_000, 8),

            # Large transfers
            (32, 2048, 8, 128, 200_000, 8),
            (32, 4096, 8, 128, 300_000, 8),
            (32, 8192, 8, 128, 500_000, 8),

            # Head slicing (mixed-TP scenario) - smaller pool to avoid OOM
            (32, 2048, 32, 128, 100_000, 8),  # 32 heads, gather 8
        ]

    # Benchmark Triton kernel
    triton_results = []
    print("\n" + "-" * 80)
    print(" Benchmarking Triton gather_kv_to_pinned kernel...")
    print("-" * 80)

    for num_layers, num_tokens, num_heads, head_dim, total_slots, num_heads_to_gather in configs:
        # Clean up memory between configs
        torch.cuda.empty_cache()

        for pattern in patterns:
            result = benchmark_gather_kv(
                num_layers=num_layers,
                num_tokens=num_tokens,
                num_heads=num_heads,
                head_dim=head_dim,
                total_slots=total_slots,
                head_start=0,
                num_heads_to_gather=num_heads_to_gather,
                pattern=pattern,
                dtype=dtype,
                warmup=args.warmup,
                rep=args.rep,
            )
            triton_results.append(result)
            print(f"  {pattern:<12} layers={num_layers:<3} tokens={num_tokens:<5} "
                  f"size={result['total_bytes_mb']:.1f}MB -> "
                  f"{result['avg_bandwidth_gbs']:.2f} GB/s (peak: {result['peak_bandwidth_gbs']:.2f})")

    print_results_table(triton_results, "Triton Zero-Copy Results")

    # Benchmark Triton with GPU staging
    staging_results = []
    print("\n" + "-" * 80)
    print(" Benchmarking Triton with GPU staging buffer...")
    print("-" * 80)

    staging_configs = [
        (4, 1024, 8, 128, 100_000, 8),
        (32, 2048, 8, 128, 200_000, 8),
        (32, 4096, 8, 128, 300_000, 8),
    ]

    for num_layers, num_tokens, num_heads, head_dim, total_slots, num_heads_to_gather in staging_configs:
        torch.cuda.empty_cache()
        for pattern in ["contiguous", "random"]:
            result = benchmark_gather_with_staging(
                num_layers=num_layers,
                num_tokens=num_tokens,
                num_heads=num_heads,
                head_dim=head_dim,
                total_slots=total_slots,
                head_start=0,
                num_heads_to_gather=num_heads_to_gather,
                pattern=pattern,
                dtype=dtype,
                warmup=args.warmup,
                rep=args.rep,
            )
            staging_results.append(result)
            print(f"  {pattern:<12} layers={num_layers:<3} tokens={num_tokens:<5} "
                  f"size={result['total_bytes_mb']:.1f}MB -> "
                  f"{result['avg_bandwidth_gbs']:.2f} GB/s (staging: {result['staging_buffer_mb']:.1f}MB)")

    print_results_table(staging_results, "Triton with GPU Staging Results")

    # Benchmark PyTorch baseline for comparison
    print("\n" + "-" * 80)
    print(" Benchmarking PyTorch baseline (gather + copy)...")
    print("-" * 80)

    baseline_results = []
    # Just test a few configs for baseline comparison
    baseline_configs = [
        (4, 1024, 8, 128, 100_000),
        (32, 2048, 8, 128, 200_000),
        (32, 4096, 8, 128, 300_000),
    ]

    for num_layers, num_tokens, num_heads, head_dim, total_slots in baseline_configs:
        for pattern in ["contiguous", "random"]:
            result = benchmark_pytorch_baseline(
                num_layers=num_layers,
                num_tokens=num_tokens,
                num_heads=num_heads,
                head_dim=head_dim,
                total_slots=total_slots,
                pattern=pattern,
                dtype=dtype,
                warmup=args.warmup,
                rep=args.rep,
            )
            result["num_layers"] = num_layers
            result["num_tokens"] = num_tokens
            baseline_results.append(result)
            print(f"  {pattern:<12} layers={num_layers:<3} tokens={num_tokens:<5} "
                  f"size={result['total_bytes_mb']:.1f}MB -> "
                  f"{result['avg_bandwidth_gbs']:.2f} GB/s")

    print_results_table(baseline_results, "PyTorch Baseline Results")

    # Benchmark raw CUDA memcpy
    print("\n" + "-" * 80)
    print(" Benchmarking raw CUDA memcpy (GPU -> Pinned CPU)...")
    print(" This shows the theoretical PCIe bandwidth ceiling.")
    print("-" * 80)

    memcpy_sizes = [16.8, 67.1, 268.4, 536.9, 1073.7]  # Match transfer sizes from above
    memcpy_results = []

    for size_mb in memcpy_sizes:
        result = benchmark_cuda_memcpy(
            size_mb=size_mb,
            dtype=dtype,
            warmup=args.warmup,
            rep=args.rep,
        )
        memcpy_results.append(result)
        print(f"  {size_mb:.1f} MB -> {result['avg_bandwidth_gbs']:.2f} GB/s (peak: {result['peak_bandwidth_gbs']:.2f})")

    print("\n" + "=" * 80)
    print(" Raw CUDA Memcpy Results (PCIe Bandwidth Ceiling)")
    print("=" * 80)
    print(f"{'Size (MB)':<12} {'Time (ms)':<12} {'BW (GB/s)':<12} {'Peak BW':<12}")
    print("-" * 48)
    for r in memcpy_results:
        print(f"{r['size_mb']:<12.1f} {r['avg_time_ms']:<12.3f} {r['avg_bandwidth_gbs']:<12.2f} {r['peak_bandwidth_gbs']:<12.2f}")

    # Summary comparison
    print("\n" + "=" * 80)
    print(" Summary: Triton vs PyTorch vs Raw Memcpy")
    print("=" * 80)

    # Get memcpy bandwidth for reference
    memcpy_bw = memcpy_results[-1]["avg_bandwidth_gbs"] if memcpy_results else 0

    print(f"\n Raw CUDA memcpy bandwidth: ~{memcpy_bw:.1f} GB/s (PCIe ceiling)")
    print("\n Bandwidth Comparison (random access pattern):")
    print(f" {'Config':<25} {'Zero-Copy':<18} {'With Staging':<18} {'PyTorch':<18}")
    print(f" {'(layers, tokens)':<25} {'GB/s (%memcpy)':<18} {'GB/s (%memcpy)':<18} {'GB/s (%memcpy)':<18}")
    print(" " + "-" * 75)

    for num_layers, num_tokens, num_heads, head_dim, total_slots in baseline_configs:
        triton_random = next(
            (r for r in triton_results
             if r["num_layers"] == num_layers and r["num_tokens"] == num_tokens
             and r["pattern"] == "random" and r.get("num_heads", num_heads) == num_heads),
            None
        )
        staging_random = next(
            (r for r in staging_results
             if r["num_layers"] == num_layers and r["num_tokens"] == num_tokens
             and r["pattern"] == "random"),
            None
        )
        pytorch_random = next(
            (r for r in baseline_results
             if r["num_layers"] == num_layers and r["num_tokens"] == num_tokens
             and r["pattern"] == "random"),
            None
        )

        config = f"({num_layers}, {num_tokens})"

        def fmt(r):
            if r is None:
                return "-"
            eff = (r['avg_bandwidth_gbs'] / memcpy_bw * 100) if memcpy_bw else 0
            return f"{r['avg_bandwidth_gbs']:.1f} ({eff:.0f}%)"

        print(f" {config:<25} {fmt(triton_random):<18} {fmt(staging_random):<18} {fmt(pytorch_random):<18}")

    # Memory tradeoff analysis
    print("\n" + "-" * 80)
    print(" GPU Memory Tradeoff Analysis:")
    print("-" * 80)
    print("\n The key advantage of Triton zero-copy is NO GPU staging buffer required.")
    print(" PyTorch baseline needs O(transfer_size) GPU memory for intermediate tensors.\n")

    memory_configs = [
        (32, 2048, 8, 128, 200_000),   # Typical
        (32, 8192, 8, 128, 500_000),   # Long context
        (80, 8192, 8, 128, 1_000_000), # Large model, long context
    ]

    print(f" {'Config':<30} {'Transfer':<12} {'PyTorch GPU':<14} {'Triton GPU':<12} {'Saved':<10}")
    print(f" {'(layers, tokens, heads)':<30} {'Size':<12} {'Staging':<14} {'Staging':<12} {'(MB)':<10}")
    print(" " + "-" * 78)

    for num_layers, num_tokens, num_heads, head_dim, total_slots in memory_configs:
        mem = benchmark_memory_usage(
            num_layers, num_tokens, num_heads, head_dim, total_slots, torch.float16
        )
        config_str = f"({num_layers}, {num_tokens}, {num_heads})"
        print(f" {config_str:<30} {mem['transfer_mb']:<12.1f} "
              f"{mem['pytorch_gpu_staging_mb']:<14.1f} {mem['triton_gpu_staging_mb']:<12.1f} "
              f"{mem['gpu_memory_saved_mb']:<10.1f}")

    print("\n For N concurrent transfers, PyTorch needs N * transfer_size GPU memory,")
    print(" while Triton needs 0 GPU staging memory (only CPU pinned buffers).")

    # Calculate example with concurrent requests
    print("\n Example: 8 concurrent 8K-token transfers (32 layers, 8 heads, 128 dim):")
    mem = benchmark_memory_usage(32, 8192, 8, 128, 500_000, torch.float16)
    concurrent = 8
    print(f"   PyTorch: {concurrent * mem['pytorch_gpu_staging_mb']:.1f} MB GPU staging required")
    print(f"   Triton:  0 MB GPU staging (zero-copy to pinned CPU)")


if __name__ == "__main__":
    main()
