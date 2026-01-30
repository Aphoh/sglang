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

| Direction | Method | Time (ms) | BW (GB/s) | % of memcpy |
|-----------|--------|-----------|-----------|-------------|
| D→H | Memcpy baseline | 512 | 12.1 | 100% |
| D→H | Zero-copy | 965 | 6.4 | 53% |
| D→H | Chunked (256MB staging) | 630 | 9.8 | **81%** |
| H→D | Memcpy baseline | 217 | 28.4 | 100% |
| H→D | Scatter (direct) | 236 | 26.2 | **92%** |

- **Gather (D→H)**: Chunked approach achieves 81% of memcpy with 256MB staging overhead
- **Scatter (H→D)**: Direct approach achieves 92% of memcpy with no staging needed

### Completed: Host → Device (Scatter) Kernel

**Location**: `python/sglang/srt/layers/attention/triton_ops/kv_transfer.py`

```python
def scatter_kv(
    pinned_input: torch.Tensor,     # pinned CPU buffer
    k_buffers: list[torch.Tensor],  # [num_layers] each [total_slots, num_heads, head_dim]
    v_buffers: list[torch.Tensor],
    slot_indices: torch.Tensor,     # [num_tokens] on GPU
    head_start: int = 0,
    num_heads_to_scatter: int = None,
) -> None
```

For the scatter kernel (pinned CPU → GPU), no staging buffer is needed because reads
from pinned memory are naturally coalesced by the GPU's memory controller.

**Benchmark Results (H100 PCIe, 40GB pool, 92 layers, 32K tokens):**

| Method | Time (ms) | BW (GB/s) | % of H→D memcpy |
|--------|-----------|-----------|-----------------|
| H→D memcpy baseline | 217 | 28.4 | 100% |
| Scatter (direct) | 236 | 26.2 | **92%** |

The scatter kernel achieves 92% of raw memcpy bandwidth with no staging buffer needed.
Reads from pinned CPU are naturally coalesced, and writes to GPU HBM are fast.

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

---

## Next Steps: Integration & Testing

### Phase 1: Understand the Dynamo Test Environment

**Location**: `~/proj/dynamo`

Dynamo is NVIDIA's inference orchestration framework that can launch disaggregated P/D sglang workers.

1. **Explore key files**:
   - `~/proj/dynamo/test_decode_disagg.py` - Existing disagg test (if present)
   - `~/proj/dynamo/lib/bindings/python/` - Python bindings for dynamo
   - `~/proj/dynamo/deploy/` - Deployment configurations
   - Look for sglang-related test files: `find ~/proj/dynamo -name "*.py" | xargs grep -l sglang`

2. **Understand the test pattern**:
   - How dynamo launches sglang workers (prefill vs decode)
   - How KV transfer is triggered between workers
   - What configuration is needed for disaggregated mode

3. **Python environment**: `~/proj/dynamo/.venv/bin/python` has sglang installed as editable

### Phase 2: Create Test Scaffolding

Create a new test file: `~/proj/dynamo/test_mixed_tp_disagg.py`

**Test Requirements**:
```python
"""
Test harness for mixed-TP disaggregated prefill/decode with Triton KV transfer.

Setup (3x A100 GPUs):
- Prefill worker: TP=2 (GPU 0,1), all heads
- Decode worker: TP=1 (GPU 2), subset of heads

Model: ~/proj/models/qwen3-4b

Test Flow:
1. Start prefill worker with TP=2
2. Start decode worker with TP=1
3. Send prefill request → triggers gather_kv on prefill side
4. NIXL transfers CPU buffer to decode worker
5. Decode worker receives → triggers scatter_kv
6. Decode generates tokens to verify correctness
"""
```

**Key test scenarios**:
1. **Basic KV transfer**: Verify data arrives correctly at decode worker
2. **Head slicing**: Prefill TP=2 sends subset of heads to decode TP=1
3. **Multi-request**: Multiple concurrent prefill→decode transfers
4. **Correctness**: Decode output matches non-disaggregated baseline

### Phase 3: Integrate Triton Kernels into sglang Disagg Path

**Files to modify in sglang**:

1. **`python/sglang/srt/disaggregation/nixl/conn.py`**:
   - Add import: `from sglang.srt.layers.attention.triton_ops.kv_transfer import gather_kv, scatter_kv`
   - Modify sender path to use `gather_kv` instead of per-slot RDMA
   - Modify receiver path to use `scatter_kv`
   - Add logging to confirm new codepath is used

2. **`python/sglang/srt/disaggregation/base/conn.py`**:
   - Check if base class needs changes for CPU buffer management

3. **Configuration**:
   - Add flag to enable/disable Triton KV transfer (for A/B testing)
   - E.g., `--kv-transfer-method triton|legacy`

### Phase 4: Add Logging for Verification

Add debug logging to verify the new codepath is executed:

```python
# In gather_kv:
import logging
logger = logging.getLogger(__name__)
logger.info(f"[TRITON-KV] gather_kv: {num_tokens} tokens, {num_layers} layers, "
            f"heads [{head_start}:{head_start+num_heads}], staging={staging_buffer is not None}")

# In scatter_kv:
logger.info(f"[TRITON-KV] scatter_kv: {num_tokens} tokens, {num_layers} layers, "
            f"heads [{head_start}:{head_start+num_heads}]")
```

**Verification checklist**:
- [ ] Log message appears during prefill worker KV send
- [ ] Log message appears during decode worker KV receive
- [ ] Transfer size matches expected (tokens × layers × heads × head_dim × 2 × dtype_size)
- [ ] Decode generates correct tokens

### Phase 5: Test Commands

```bash
# Activate dynamo environment
source ~/proj/dynamo/.venv/bin/activate

# Run the test (adjust based on dynamo's test runner)
python ~/proj/dynamo/test_mixed_tp_disagg.py \
    --model ~/proj/models/qwen3-4b \
    --prefill-tp 2 \
    --decode-tp 1 \
    --num-gpus 3

# Or if dynamo uses pytest:
pytest ~/proj/dynamo/test_mixed_tp_disagg.py -v -s
```

### Phase 6: Debug Workflow

If KV transfer fails:

1. **Check logs for `[TRITON-KV]`** - confirms new codepath
2. **Verify pinned buffer allocation** - must be page-aligned
3. **Check slot_indices** - must be valid indices into KV cache
4. **NIXL transfer completion** - ensure CPU→CPU transfer completes before scatter
5. **Head index math** - verify head_start/num_heads matches TP configuration

### Open Questions to Resolve

1. **How does dynamo configure TP for different workers?**
   - Need to find where TP is specified per-worker

2. **How is NIXL connection established between workers?**
   - Look for nixl initialization in dynamo/sglang integration

3. **What's the buffer handoff protocol?**
   - When does prefill signal "buffer ready"?
   - When does decode know transfer is complete?

4. **Head assignment for mixed TP**:
   - Prefill TP=2: Worker 0 has heads [0:H/2], Worker 1 has heads [H/2:H]
   - Decode TP=1: Single worker needs all heads
   - Each prefill worker sends its heads to decode worker

### File Checklist

| File | Status | Description |
|------|--------|-------------|
| `kv_transfer.py` | ✅ Done | Triton gather/scatter kernels |
| `test_kv_transfer.py` | ✅ Done | Unit tests for kernels |
| `bench_kv_transfer.py` | ✅ Done | Bandwidth benchmarks |
| `nixl/conn.py` | ⬜ TODO | Integrate gather/scatter into disagg path |
| `test_mixed_tp_disagg.py` | ⬜ TODO | Dynamo integration test |
| Logging | ⬜ TODO | Add [TRITON-KV] logging for verification |
