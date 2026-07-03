# Qwen3 TP2 CUDA graph checkpoint and restore

This experiment checkpoints a dense Qwen3 TP=2 engine without an external
collective package or a patched FlashInfer build. Both checkpoint-sensitive
collectives are owned by SGLang:

- `sgl-kernel` custom all-reduce, using renewable VMM-backed signal and staging
  buffers during checkpoint mode;
- `sgl-kernel` symmetric-memory all-gather, ported from the prior FlashInfer
  workspace onto a renewable peer-mapped VMM buffer.

The CUDA graph retains the same virtual addresses across restore. Before CRIU,
SGLang synchronizes CUDA, unmaps and releases collective backing allocations,
and destroys its Gloo/NCCL process groups. After restore it recreates the
groups, creates new physical allocations, maps them at the reserved addresses,
resets protocol state, and replays the original graph.

## Validated scope

- dense Qwen3 BF16, TP=2;
- one pre-checkpoint decode graph replayed after restore;
- restartable Gloo control groups and the default NCCL device group;
- optional complete physical-GPU UUID remapping on driver 580 or newer.

Checkpoint mode rejects PP, DP, EP, CP, MoE, disaggregation, HiCache, HiSparse,
radix-cache checkpointing, the separate `--enable-symm-mem` backend,
FlashInfer all-reduce fusion, and owned non-WORLD device subgroups.

## All-gather performance cost

Qwen3 performs this all-gather once per normal decode forward, when the logits
processor gathers the TP-sharded LM-head logits. For Qwen3-4B TP=2, the exact
per-rank BF16 payload is 151,936 bytes. On two B200s, the slowest-rank median
was 24.61 us for the checkpointable symmetric all-gather and 12.67 us for NCCL
`all_gather_into_tensor`.

The validated single-request decode run produced 245.48 tokens/s, or 4.07 ms
per token. Relative to that observed decode/model-forward step:

- the **entire checkpointable all-gather is about 0.60%** of the step;
- NCCL itself would be about 0.31%; and
- the **incremental checkpointability cost versus NCCL is about 0.29%**
  (11.94 us per token).

Thus the custom all-gather is well below 1% of the measured Qwen3 decode path,
and the performance premium paid for the renewable VMM/checkpoint lifecycle is
roughly three-tenths of one percent. These figures are for BF16, TP=2,
decode batch size 1 on B200 and should be remeasured for other workloads.

## Build and run

```bash
docker build \
  -f examples/experimental/criu/Dockerfile \
  -t sglang-native-criu:latest .

docker run --rm --privileged --gpus all \
  -v "$PWD:/workspace/sglang" \
  -v /tmp/sglang-checkpoints:/checkpoint/sglang \
  sglang-native-criu:latest \
  bash examples/experimental/criu/run_qwen3_tp2_criu.sh
```

The runner builds the in-tree `sgl-kernel` before launch. Its CMake/object cache
defaults to the git-ignored `sgl-kernel/build`; set `SGLANG_KERNEL_BUILD_DIR`
to place the persistent cache elsewhere. A cold run installs the full package.
Warm runs load the in-tree Python package against the cached `common_ops`
artifacts, avoiding wheel repackaging; Ninja runs only when native kernel
sources are newer than those libraries. The CRIU build disables the unrelated
FA3 extension while retaining the common SGLang ops used by the model. Useful
overrides include:

- `SGLANG_CRIU_MODEL` (default `Qwen/Qwen3-4B`)
- `SGLANG_CRIU_GPUS` (default `0,1`)
- `SGLANG_CRIU_ALL_REDUCE_MAX_BYTES` (default 32 MiB)
- `SGLANG_CRIU_ALL_GATHER_MAX_ELEMS` (default 262144)
- `SGLANG_CRIU_DEVICE_MAP` (`old-uuid=new-uuid` pairs)
- `SGLANG_CRIU_SOURCE_GPU_UUIDS` and `SGLANG_CRIU_RESTORE_GPU_UUIDS`

The local CRIU plugin only reopens `/dev/nvidiactl`; NVIDIA's
`cuda-checkpoint-helper` handles CUDA state. Results and before/after GPU
residency are written under the printed checkpoint state directory.
