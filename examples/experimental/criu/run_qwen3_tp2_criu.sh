#!/usr/bin/env bash
set -euo pipefail

root=${SGLANG_CRIU_STATE:-/checkpoint/sglang}
timeout_seconds=${SGLANG_CRIU_TIMEOUT:-900}
criu_timeout_seconds=${SGLANG_CRIU_OPERATION_TIMEOUT:-120}
criu_plugin_dir=${CRIU_PLUGIN_DIR:-/usr/local/lib/criu}
cuda_checkpoint=${CUDA_CHECKPOINT_BIN:-/usr/local/sbin/cuda-checkpoint}
mkdir -p "${root}"
state=$(mktemp -d "${root%/}/run.XXXXXX")
run_dir="${state}/run"
images_dir="${state}/images"
mkdir -p "${run_dir}" "${images_dir}"
echo "checkpoint state: ${state}"

criu --version
criu check --libdir "${criu_plugin_dir}"

"${cuda_checkpoint}" --launch-job \
  env NCCL_IB_DISABLE=1 UCX_TLS=cuda_ipc,cuda_copy,sm,self \
  UV_USE_IO_URING=0 USE_LIBUV=0 \
  python3 /opt/sglang/examples/experimental/criu/qwen3_tp2.py \
    --model "${SGLANG_CRIU_MODEL:-Qwen/Qwen3-4B}" \
    --rendezvous "${run_dir}" \
    --tp-size "${SGLANG_CRIU_TP_SIZE:-2}" \
    --timeout "${timeout_seconds}" \
    --mem-fraction-static "${SGLANG_CRIU_MEM_FRACTION_STATIC:-0.1}" \
    --max-total-tokens "${SGLANG_CRIU_MAX_TOTAL_TOKENS:-4096}" \
    >"${state}/engine.log" 2>&1 &
launcher_pid=$!

wait_for() {
  local path=$1 description=$2
  local deadline=$((SECONDS + timeout_seconds))
  while [[ ! -e "${path}" ]]; do
    local monitored_pid=${controller_pid:-${launcher_pid}}
    if ! kill -0 "${monitored_pid}" 2>/dev/null; then
      echo "engine exited while waiting for ${description}" >&2
      cat "${state}/engine.log" >&2
      exit 1
    fi
    if ((SECONDS >= deadline)); then
      echo "timed out waiting for ${description}" >&2
      exit 1
    fi
    sleep 0.1
  done
}

cleanup() {
  local status=$?
  if ((status != 0)); then
    cat "${state}/engine.log" >&2 || true
    cat "${images_dir}/dump.log" >&2 2>/dev/null || true
    cat "${images_dir}/restore.log" >&2 2>/dev/null || true
  fi
  if ((status != 0)); then
    kill "${controller_pid:-${launcher_pid}}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT

wait_for "${run_dir}/job-ready" "SGLang suspend"
controller_pid=$(<"${run_dir}/controller-pid")
controller_start_time=$(awk '{print $22}' "/proc/${controller_pid}/stat")
mapfile -t candidates < <(jq -r '.[]' "${run_dir}/worker-pids.json")
cuda_pids=()
for pid in "${candidates[@]}"; do
  if "${cuda_checkpoint}" --get-restore-tid --pid "${pid}" >/dev/null 2>&1; then
    cuda_pids+=("${pid}")
  fi
done
if [[ ${#cuda_pids[@]} -lt 2 ]]; then
  echo "expected at least two CUDA worker processes, found ${#cuda_pids[@]}" >&2
  exit 1
fi
echo "CUDA worker PIDs: ${cuda_pids[*]}"

timeout "${timeout_seconds}s" criu dump \
  --tree "${controller_pid}" \
  --images-dir "${images_dir}" \
  --shell-job --file-locks --link-remap --tcp-established --skip-in-flight \
  --manage-cgroups=ignore --ghost-limit 128M \
  --timeout "${criu_timeout_seconds}" \
  --libdir "${criu_plugin_dir}" \
  --log-file dump.log -v4

timeout "${timeout_seconds}s" criu restore \
  --images-dir "${images_dir}" \
  --shell-job --file-locks --link-remap --tcp-established \
  --manage-cgroups=ignore --restore-detached \
  --timeout "${criu_timeout_seconds}" \
  --libdir "${criu_plugin_dir}" \
  --log-file restore.log -v4

for pid in "${cuda_pids[@]}"; do
  state_after_restore=$("${cuda_checkpoint}" --get-state --pid "${pid}")
  if [[ "${state_after_restore}" != running ]]; then
    echo "CUDA worker ${pid} is ${state_after_restore}, expected running" >&2
    exit 1
  fi
done

touch "${run_dir}/resume"
wait_for "${run_dir}/passed" "post-restore generation"

deadline=$((SECONDS + timeout_seconds))
while [[ -r "/proc/${controller_pid}/stat" ]] \
    && [[ "$(awk '{print $22}' "/proc/${controller_pid}/stat")" == "${controller_start_time}" ]]; do
  if ((SECONDS >= deadline)); then
    echo "timed out waiting for restored controller to exit" >&2
    exit 1
  fi
  sleep 0.1
done
launcher_status=0
wait "${launcher_pid}" || launcher_status=$?
# The original tree dies during dump; bash can retain that SIGKILL status even
# after the restored process with the same PID exits successfully.
if [[ ${launcher_status} -ne 0 && ${launcher_status} -ne 137 ]]; then
  echo "cuda-checkpoint launcher exited with status ${launcher_status}" >&2
  exit 1
fi

cat "${run_dir}/result.json"
cat "${state}/engine.log"
echo "Qwen3 TP2 CUDA/CRIU checkpoint and restore passed"
