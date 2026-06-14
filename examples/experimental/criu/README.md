# Qwen3 TP2 CUDA graph checkpoint and restore

This branch adds a deliberately scoped dense-TP2 CRIU lifecycle to SGLang and consumes the
[`movin`](https://github.com/NVIDIA-dev/warnold-movin) Python package for
checkpointable collective implementations.

The authoritative handoff recipe lives in Movin:

<https://github.com/NVIDIA-dev/warnold-movin/tree/warnold/all-gather-kernel/recipes/sglang_qwen3_tp2_criu>

## Collective configuration

The validated Qwen3-8B TP2 configuration uses:

- FlashInfer TRT-LLM raw and fused all-reduce;
- Movin's symmetric-VMM BF16 all-gather;
- Gloo only for restartable control-plane handle exchange;
- no retained NCCL communicator during checkpoint;
- the same captured decode graph before and after restore.

This experiment does not yet claim support for PP/DP device subgroups, MoE
all-to-all, disaggregation, HiCache, or radix-cache configurations that retain
raw process-group references. A checkpoint preflight rejects owned non-WORLD
device groups before detaching any CUDA resources.

SGLang owns framework orchestration. Movin owns CUDA/NIXL/FlashInfer workspace
lifecycle and kernel wrappers. `GroupCoordinator` exposes one
`movin.CollectiveSet`; CRIU ordering is owned by
`CriuCheckpointCoordinator`.

## Build

```bash
docker build \
  -f examples/experimental/criu/Dockerfile \
  -t sglang-movin-criu:handoff .
```

The digest-pinned image contains the patched CRIU fork, NVIDIA CUDA checkpoint tool, and
checkpoint helper. The runtime runner installs a mounted Movin checkout as an
editable package and explicitly fetches its pinned NIXL/UCX build headers.

## Direct run

The Movin recipe is preferred because it pins inputs, verifies results, archives
logs, and cleans checkpoint images. For local development:

```bash
docker run --rm \
  --privileged --pid=host --ipc=host \
  --gpus 'device=4,5' \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e SGLANG_CRIU_GPUS=0,1 \
  -e SGLANG_CRIU_MODEL=/mnt/shared/Qwen3-8B \
  -e SGLANG_CRIU_GSM8K_QUESTIONS=200 \
  -e SGLANG_CRIU_GSM8K_MAX_TOKENS=8192 \
  -e SGLANG_CRIU_MAX_TOTAL_TOKENS=16384 \
  -e SGLANG_CRIU_GSM8K_MIN_ACCURACY=0.90 \
  -e SGLANG_TP_ALL_REDUCE_BACKEND=flashinfer \
  -e MOVIN_REPO=/workspace/movin \
  -e SGLANG_CRIU_STATE=/checkpoint/sglang \
  -v /path/to/warnold-movin:/workspace/movin \
  -v "$PWD":/workspace/sglang \
  -v /mnt/shared:/mnt/shared:ro \
  -v /path/to/empty-checkpoint-state:/checkpoint/sglang \
  -w /workspace/sglang \
  sglang-movin-criu:handoff \
  bash examples/experimental/criu/run_qwen3_tp2_criu.sh
```

## Validated result

Hardware and software:

- two NVIDIA B200 GPUs, 183,359 MiB each;
- NVIDIA driver 595.58.03;
- CUDA image 13.0.1;
- PyTorch 2.11.0+cu130;
- FlashInfer 0.6.11.post1;
- NIXL 1.1.0;
- CRIU 4.2 at commit `00b4a49`;
- SGLang 0.5.12.post1 base image.

Workload:

- Qwen3-8B BF16, TP2;
- 200 GSM8K questions, five-shot;
- 8192 maximum new tokens;
- 16384 maximum total tokens;
- batch-size-1 decode CUDA graph.

| Metric | Before checkpoint | After restore |
|---|---:|---:|
| GSM8K accuracy | 0.93 | 0.93 |
| Output throughput | 232.127 token/s | 233.698 token/s |
| Completion tokens | 28056 | 28056 |
| Invalid rate | 0 | 0 |

All 200 predictions and all 200 generated texts were identical.

The matching all-gather microbenchmark measured 21.410 microseconds kernel-only
and 24.530 microseconds end to end, versus the previous 5.649 millisecond forced
NIXL-transfer fallback.

## Lifecycle

1. Capture graph-visible collective workspaces.
2. Quiesce CUDA.
3. Detach FlashInfer and Movin physical backing/registrations.
4. Tear down CPU and device process groups.
5. Checkpoint CUDA and dump the process tree with CRIU.
6. Restore the process tree and CUDA.
7. Recreate process groups and handle-exchange backends.
8. Remap fresh physical storage at graph-stable virtual addresses.
9. Replay the original graph and compare exact output.

Checkpoint mode fails immediately if a configured collective cannot cover a
tensor shape; it does not silently fall through to a Gloo CUDA operation.
