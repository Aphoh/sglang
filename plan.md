# Mixed-TP KV Cache Transfer via Direct GPU-to-Pinned-CPU Triton Kernels

## Overview

Replace the current per-token-slot RDMA descriptor approach with:
1. **Triton kernel**: Gather scattered KV data + slice heads → write to pinned CPU buffer
2. **CPU-to-CPU transfer**: Via NIXL (or TCP fallback)
3. **Triton kernel**: Read from pinned CPU buffer → scatter to KV cache

This reduces RDMA descriptor count from O(tokens × layers) to O(1) per transfer.

## Implementation Status

### Completed: Device → Host (Gather) Kernel

**Location**: `python/sglang/srt/layers/attention/triton_ops/kv_transfer.py`

Primary API:
```python
def gather_kv(
    k_buffers: list[torch.Tensor],  # [num_layers] each [total_slots, num_heads, head_dim]
    v_buffers: list[torch.Tensor],
    slot_indices: torch.Tensor,     # [num_tokens] on GPU
    pinned_output: torch.Tensor,    # pinned CPU buffer
    head_start: int = 0,
    num_heads_to_gather: int = None,
    staging_buffer: torch.Tensor = None,
    staging_buffer_size_mb: float = 256.0,
) -> None
```

**Key Finding: Chunked Staging Approach**

Direct zero-copy writes to pinned CPU memory achieve only ~50% of PCIe bandwidth due to
uncoalesced writes over PCIe. The solution is a **chunked staging buffer approach**:

1. Allocate a fixed-size GPU staging buffer (default 256MB)
2. Gather scattered KV data into contiguous GPU staging buffer (fast coalesced writes)
3. Copy from GPU staging to pinned CPU (optimized memcpy)
4. Repeat for remaining data

**Benchmark Results (H100 PCIe, 40GB pool, 92 layers, 32K tokens, 6.17GB transfer):**

| Method | Time (ms) | BW (GB/s) | % of PCIe |
|--------|-----------|-----------|-----------|
| Raw memcpy | 512 | 12.06 | 100% |
| Zero-copy | 965 | 6.40 | **53%** |
| Chunked (256MB staging) | 630 | 9.79 | **81%** |

The chunked approach achieves 81% of PCIe bandwidth while limiting GPU memory overhead
to only 256MB regardless of transfer size.

### TODO: Host → Device (Scatter) Kernel

For the scatter kernel (pinned CPU → GPU), we don't need the staging buffer approach
because reads from pinned memory are naturally coalesced by the GPU's memory controller.
The kernel can read directly from pinned CPU and write scattered to GPU KV cache.

## Architecture

```
Sender (Prefill):
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  KV Cache   │───▶│ GPU Staging │───▶│ Pinned CPU  │───▶ NIXL
│  (scattered)│    │  (256MB)    │    │   Buffer    │
└─────────────┘    └─────────────┘    └─────────────┘
         Triton gather      memcpy (81% PCIe BW)

Receiver (Decode):
        ┌─────────────┐         ┌─────────────┐
NIXL ──▶│ Pinned CPU  │────────▶│  KV Cache   │
        │   Buffer    │         │  (scattered)│
        └─────────────┘         └─────────────┘
         Triton scatter (direct read, no staging needed)
```

## File Structure

| File | Description |
|------|-------------|
| `python/sglang/srt/layers/attention/triton_ops/kv_transfer.py` | Triton kernels for gather/scatter |
| `sgl-kernel/tests/test_kv_transfer.py` | Correctness tests |
| `sgl-kernel/benchmarks/bench_kv_transfer.py` | Bandwidth benchmarks |

## Key Insights

1. **Pinned memory is GPU-accessible**: Triton kernels can read/write pinned CPU memory
   directly via zero-copy, but writes are slow due to PCIe characteristics.

2. **Coalesced writes matter**: Writing scattered data directly to pinned CPU achieves
   only ~50% of PCIe bandwidth. Using a GPU staging buffer for coalesced writes first,
   then memcpy to pinned, achieves ~80%.

3. **Reads are different**: Reading from pinned CPU to GPU doesn't have the same issue
   because the GPU's memory controller can coalesce the reads.

4. **Fixed staging buffer size**: 256MB staging buffer provides nearly optimal bandwidth
   regardless of transfer size, with diminishing returns beyond that.

## Integration Points

To integrate with existing code in `python/sglang/srt/disaggregation/nixl/conn.py`:

```python
from sglang.srt.layers.attention.triton_ops.kv_transfer import gather_kv, scatter_kv

# Sender side
gather_kv(
    k_buffers=self.k_buffer,
    v_buffers=self.v_buffer,
    slot_indices=torch.from_numpy(prefill_kv_indices).to(self.device),
    pinned_output=self.cpu_send_buffer,
    head_start=head_start,
    num_heads_to_gather=num_heads_to_send,
)

# Single NIXL transfer (CPU → CPU) instead of O(tokens × layers) descriptors
# ...

# Receiver side (after NIXL completes)
scatter_kv(
    pinned_input=self.cpu_recv_buffer,
    k_buffers=self.k_buffer,
    v_buffers=self.v_buffer,
    slot_indices=torch.from_numpy(dst_kv_indices).to(self.device),
    head_start=head_start,
    num_heads_received=num_heads_received,
)
```

## Performance Summary

| Metric | Current Approach | New Approach |
|--------|-----------------|--------------|
| NIXL descriptors | O(tokens × layers × page_size) | O(1) |
| GPU staging memory | None | 256MB fixed |
| PCIe efficiency | N/A (RDMA per slot) | ~80% of bandwidth |
| CPU orchestration | High (Python loops) | Minimal |
