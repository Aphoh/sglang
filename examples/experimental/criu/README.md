# Qwen3 TP2 CUDA graph checkpoint and restore

This branch adds a scoped dense-TP checkpoint lifecycle to SGLang and uses
[`movin`](https://github.com/NVIDIA-dev/warnold-movin) for checkpointable
collective workspaces.

The authoritative, pinned reproduction lives in Movin:

<https://github.com/NVIDIA-dev/warnold-movin/tree/warnold/all-gather-kernel/recipes/sglang_qwen3_tp2_criu>

## Validated Configuration

- Qwen3-8B BF16, TP2
- FlashInfer TRT-LLM raw and fused all-reduce
- Movin symmetric-VMM BF16 all-gather
- restartable Gloo control groups
- one decode graph captured before checkpoint and replayed after restore

The SGLang coordinator quiesces CUDA, detaches collective resources, destroys
CPU and device process groups, waits for external checkpoint/restore, recreates
the groups and collective backing, then releases the workers.

The shell runner uses NVIDIA's external `cuda-checkpoint-helper`. It discovers
CUDA-bearing PIDs and explicitly performs:

1. CUDA lock and checkpoint;
2. CRIU process-tree dump and restore;
3. CUDA restore and unlock.

On display driver 580 or newer, the runner also accepts a complete
`SGLANG_CRIU_DEVICE_MAP` of `old-uuid=new-uuid` pairs. The Movin handoff recipe
uses this to expose a source and destination pair, checkpoint TP2 on the source,
and restore the same logical CUDA devices and captured graph onto the
destination. On the validated privileged-container setup, the map is a
host-wide bijection: selected source and destination UUIDs are swapped and all
other physical GPUs are identity-mapped. The runner applies the map only to
CUDA processes resident on source GPUs. `SGLANG_CRIU_SOURCE_GPU_UUIDS` and
`SGLANG_CRIU_RESTORE_GPU_UUIDS` enable strict NVML residency checks around the
checkpoint.

The CRIU `nvidiactl` plugin compiled by the runner only reopens
`/dev/nvidiactl` descriptors. It does not checkpoint CUDA state.

## Build

```bash
docker build \
  -f examples/experimental/criu/Dockerfile \
  -t sglang-movin-criu:handoff .
```

For standalone package installation:

```bash
uv pip install -e './python[criu]'
```

The Movin recipe instead mounts an exact Movin checkout and records both git
commits in the experiment provenance.

## Reference Result

The June 14, 2026 reference run used two NVIDIA B200 GPUs, driver 595.58.03,
CUDA 13.0.1, PyTorch 2.11.0+cu130, FlashInfer 0.6.11.post1, NIXL 1.1.0, and
CRIU commit `00b4a49`.

| Metric | Before checkpoint | After restore |
|---|---:|---:|
| GSM8K accuracy | 0.93 | 0.93 |
| Output throughput | 232.725 token/s | 234.023 token/s |

All 200 predictions and generated texts were identical.

## Scope

Checkpoint mode rejects unsupported configurations before detaching resources.
The validated scope excludes PP, DP, MoE all-to-all, disaggregation, HiCache,
radix-cache checkpointing, and owned non-WORLD device subgroups.
