# Qwen3 TP2 CUDA/CRIU checkpoint and restore

This recipe exercises the checkpoint lifecycle without an external collective
package. SGLang uses its renewable sgl-kernel all-reduce and all-gather,
destroys Gloo and NCCL process groups, and waits while the host checkpoints the
CUDA worker processes and the full process tree.

## Scope

The validated runtime scope is dense, single-node TP with PP=DP=EP=CP=1. The
server must be idle at suspend. The example uses the same physical GPUs after
restore and disables InfiniBand transports; GPU and network migration are
intentionally outside this minimal recipe.

Prerequisites:

- NVIDIA display driver 570 or newer
- a privileged container with the NVIDIA runtime
- a shared writable checkpoint directory
- enough host memory for CUDA checkpoint state and CRIU images

## Build

The image pins the same CRIU and NVIDIA cuda-checkpoint revisions used by the
earlier prototype, then builds sgl-kernel from this checkout. The pinned CRIU
fork installs its CUDA plugin; the runner launches the tree with
`cuda-checkpoint --launch-job`, and CRIU drives the CUDA lock/checkpoint and
restore/unlock hooks. A tiny companion plugin reopens `/dev/nvidiactl` and
accepts inactive NVIDIA device mappings left by NVML in non-CUDA controller
processes. Docker layer, uv, pip, and CMake caches are retained between
iterations. On the validation host, a clean focused sgl-kernel build took
5m31s; source-only rebuilds reused those caches and took 9--17s.

```bash
docker build \
  -f examples/experimental/criu/Dockerfile \
  -t sglang-qwen3-criu .
```

## Run

```bash
docker run --rm --privileged \
  --gpus all --ipc=host --network=host \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e SGLANG_CRIU_MODEL=Qwen/Qwen3-4B \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v /var/tmp/sglang-checkpoints:/checkpoint \
  sglang-qwen3-criu \
  bash /opt/sglang/examples/experimental/criu/run_qwen3_tp2_criu.sh
```

The workload generates a deterministic completion, suspends SGLang, and writes
the worker PID manifest. The host script locks and checkpoints every CUDA
worker, dumps/restores the process tree with CRIU, restores CUDA, resumes
SGLang, and requires byte-identical post-restore text.

## Runtime cost

The checkpoint mode reserves about 240 MiB/GPU at TP2: roughly 192 MiB for the
64 MiB all-reduce metadata, staging, and pointer-registration allocations and
48 MiB for triple-buffered 8 MiB all-gather slots.

At Qwen3-4B's 151,936-byte TP2 decode payload, the checkpointable all-gather
takes 22.52 us in an isolated CUDA graph versus 16.02 us for NCCL. That is
0.55% of a measured 4.07 ms decode step, with a 0.16% incremental cost over
NCCL.

One full B200 validation run measured:

| Phase | Time |
|---|---:|
| Warm 8-token request before checkpoint | 55.7 ms |
| SGLang suspend | 976 ms |
| CRIU dump | 132.9 s |
| CRIU restore | 91.4 s |
| Total external checkpoint pause | 230.6 s |
| SGLang resume | 157 ms |
| Warm 8-token request after restore | 41.5 ms |

The post-restore text was byte-identical. The request-latency difference is
normal warm-run variance, not an expected speedup; end-to-end downtime is
dominated by serializing and restoring CUDA/process state, not by the custom
collective or SGLang lifecycle.
