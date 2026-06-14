#!/usr/bin/env bash
set -euo pipefail

probe_bin=${CUDA_CHECKPOINT_PROBE_BIN:-/usr/local/bin/cuda-checkpoint-helper}
action_bin=${CUDA_CHECKPOINT_ACTION_BIN:-/usr/local/sbin/cuda-checkpoint}

case "${1:-}" in
  --get-restore-tid|--get-state|-h|--help)
    exec "${probe_bin}" "$@"
    ;;
  --action)
    exec "${action_bin}" "$@"
    ;;
  *)
    exec "${action_bin}" "$@"
    ;;
esac
