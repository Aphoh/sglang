#!/usr/bin/env bash
set -euo pipefail

sglang_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
movin_root=${MOVIN_REPO:-/workspace/movin}
state_root=${SGLANG_CRIU_STATE:-/checkpoint/sglang}
timeout_seconds=${SGLANG_CRIU_TIMEOUT:-900}
criu_timeout_seconds=${SGLANG_CRIU_OPERATION_TIMEOUT:-120}
python_bin=${SGLANG_PYTHON:-/usr/bin/python}
uv_bin=${UV_BIN:-uv}
movin_deps_dir=${MOVIN_DEPS_DIR:-/var/cache/movin/deps}
cuda_checkpoint_helper=${CUDA_CHECKPOINT_HELPER_BIN:-/usr/local/bin/cuda-checkpoint-helper}
cuda_device_map=${SGLANG_CRIU_DEVICE_MAP:-}
source_gpu_uuids=${SGLANG_CRIU_SOURCE_GPU_UUIDS:-}
restore_gpu_uuids=${SGLANG_CRIU_RESTORE_GPU_UUIDS:-${source_gpu_uuids}}
host_tools="${sglang_root}/examples/experimental/criu/criu_host_tools.py"
mkdir -p "${state_root}"
state_dir=$(mktemp -d "${state_root%/}/run.XXXXXX")
run_dir="${state_dir}/run"
images_dir="${state_dir}/images"
plugin_dir="${state_dir}/plugins"
dist_store="${run_dir}/torch-dist-store"
mkdir -p "${run_dir}" "${images_dir}" "${plugin_dir}"

expected_movin_commit=${MOVIN_EXPECTED_COMMIT:-}
actual_movin_commit=${MOVIN_ACTUAL_COMMIT:-}
if [[ -z "${expected_movin_commit}" || -z "${actual_movin_commit}" ]]; then
  expected_movin_commit=$(
    "${python_bin}" "${host_tools}" pinned-movin-commit \
      "${sglang_root}/python/pyproject.toml"
  )
  actual_movin_commit=$(git -C "${movin_root}" rev-parse HEAD)
  if [[ "${actual_movin_commit}" != "${expected_movin_commit}" ]]; then
    if ! git -C "${movin_root}" merge-base --is-ancestor \
        "${expected_movin_commit}" "${actual_movin_commit}" \
        || ! git -C "${movin_root}" diff --quiet \
          "${expected_movin_commit}" "${actual_movin_commit}" -- \
          pyproject.toml uv.lock python kernels; then
      echo "Movin package inputs differ from pin ${expected_movin_commit}" >&2
      exit 1
    fi
  fi
  if [[ -n "$(git -C "${movin_root}" status --porcelain)" ]]; then
    echo "Movin checkout must be clean for a reproducible run" >&2
    exit 1
  fi
elif [[ "${MOVIN_PACKAGE_INPUTS_VALIDATED:-}" != 1 ]]; then
  echo "Host-provided Movin commits require validated package inputs" >&2
  exit 1
fi
echo "Movin package pin: ${expected_movin_commit}; checkout: ${actual_movin_commit}"
"${uv_bin}" pip install --system --break-system-packages --no-deps \
  --editable "${movin_root}" \
  --editable "${sglang_root}/python"
MOVIN_DEPS_DIR="${movin_deps_dir}" "${python_bin}" -m movin.build_deps
"${python_bin}" -c 'import flashinfer, importlib.metadata as m, movin, sglang; assert m.version("flashinfer-python") == "0.6.13"; assert flashinfer.__git_version__ == "1ac01069632461c4a84110d2c9c630a95a4c77e3"; assert m.version("flashinfer-cubin") == "0.6.13"; print("movin package:", movin.__file__); print("sglang package:", sglang.__file__)'

cc -O2 -Wall -Wextra -Werror -shared -fPIC \
  "${movin_root}/examples/cuda_checkpoint/nvidiactl_criu_plugin.c" \
  -o "${plugin_dir}/nvidiactl_plugin.so"

echo "checkpoint state: ${state_dir}"
criu --version
criu check

controller_log="${state_dir}/controller.log"
(
  cd "${sglang_root}"
  exec /usr/local/sbin/cuda-checkpoint --launch-job \
    env \
      NCCL_IB_DISABLE=1 \
      UCX_TLS=cuda_ipc,cuda_copy,sm,self \
      UV_USE_IO_URING=0 \
      MOVIN_DEPS_DIR="${movin_deps_dir}" \
      USE_LIBUV=0 \
      SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE="file://${dist_store}" \
      SGLANG_CRIU_REUSE_DEVICE_PROCESS_GROUPS=1 \
      SGLANG_CRIU_DISABLE_PYNCCL=1 \
      SGLANG_CRIU_DISABLE_TORCH_NCCL=1 \
      SGLANG_CRIU_SUSPEND_DEVICE_PROCESS_GROUP=1 \
      SGLANG_CRIU_DEVICE_STORE="${run_dir}/torch-device-store" \
      SGLANG_MOVIN_NIXL_MAX_ELEMS=262144 \
      SGLANG_MOVIN_NIXL_ALLGATHER_MAX_ELEMS=262144 \
      SGLANG_MOVIN_NIXL_ENABLE_ALLGATHER=1 \
      SGLANG_TP_ALL_REDUCE_BACKEND="${SGLANG_TP_ALL_REDUCE_BACKEND:-flashinfer}" \
      "${python_bin}" examples/experimental/criu/qwen3_tp2_movin.py \
        --model "${SGLANG_CRIU_MODEL:-Qwen/Qwen3-4B}" \
        --checkpoint-backend external-criu \
        --rendezvous "${run_dir}" \
        --gpus "${SGLANG_CRIU_GPUS:-0,1}" \
        --tp-size "${SGLANG_CRIU_TP_SIZE:-2}" \
        --timeout "${timeout_seconds}" \
        --gsm8k-data-path "${SGLANG_CRIU_GSM8K_DATA:-test.jsonl}" \
        --gsm8k-num-questions "${SGLANG_CRIU_GSM8K_QUESTIONS:-1}" \
        --gsm8k-num-shots "${SGLANG_CRIU_GSM8K_SHOTS:-5}" \
        --gsm8k-max-new-tokens "${SGLANG_CRIU_GSM8K_MAX_TOKENS:-64}" \
        --gsm8k-min-accuracy "${SGLANG_CRIU_GSM8K_MIN_ACCURACY:-0.0}" \
        --max-total-tokens "${SGLANG_CRIU_MAX_TOTAL_TOKENS:-4096}" \
        --mem-fraction-static "${SGLANG_CRIU_MEM_FRACTION_STATIC:-0.60}" \
        ${SGLANG_CRIU_DISABLE_FLASHINFER_FUSION:+--disable-flashinfer-allreduce-fusion}
) >"${controller_log}" 2>&1 &
launcher_pid=$!

wait_for_file() {
  local path=$1
  local description=$2
  local deadline=$((SECONDS + timeout_seconds))
  while [[ ! -e "${path}" ]]; do
    if ! kill -0 "${launcher_pid}" 2>/dev/null; then
      echo "controller exited while waiting for ${description}" >&2
      cat "${controller_log}" >&2
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "timed out waiting for ${description}" >&2
      cat "${controller_log}" >&2
      exit 1
    fi
    sleep 0.1
  done
}

capture_worker_gpu_residency() {
  local phase=$1
  local expected_gpu_uuids=$2
  "${python_bin}" "${host_tools}" record-gpu-residency \
    "${phase}" \
    "${run_dir}/worker-pids.json" \
    "${expected_gpu_uuids}" \
    "${run_dir}/gpu-migration.json"
}

wait_for_worker_gpu_residency() {
  local phase=$1
  local expected_gpu_uuids=$2
  local error_log="${run_dir}/gpu-residency-${phase}.error"
  local deadline=$((SECONDS + 30))
  while ! capture_worker_gpu_residency \
      "${phase}" "${expected_gpu_uuids}" 2>"${error_log}"; do
    if (( SECONDS >= deadline )); then
      cat "${error_log}" >&2
      return 1
    fi
    sleep 0.1
  done
  rm -f "${error_log}"
}

wait_for_file "${run_dir}/job-ready" "SGLang checkpoint barrier"
controller_pid=$(<"${run_dir}/controller-pid")

process_start_time() {
  "${python_bin}" "${host_tools}" process-start-time "$1"
}

wait_for_process_exit() {
  local pid=$1
  local start_time=$2
  local deadline=$((SECONDS + timeout_seconds))
  local current
  while current=$(process_start_time "${pid}" 2>/dev/null) \
      && [[ "${current}" == "${start_time}" ]]; do
    if (( SECONDS >= deadline )); then
      echo "timed out waiting for restored controller ${pid} to exit" >&2
      return 1
    fi
    sleep 0.1
  done
}

controller_start_time=$(process_start_time "${controller_pid}")

locked=()
checkpointed=()
declare -A pid_start_times=()

same_process() {
  local pid=$1
  local expected=${pid_start_times["${pid}"]:-}
  local current
  [[ -n "${expected}" ]] || return 1
  current=$(process_start_time "${pid}" 2>/dev/null) || return 1
  [[ "${current}" == "${expected}" ]]
}

cleanup_action() {
  local action=$1
  local pid=$2
  if same_process "${pid}"; then
    timeout 10s "${cuda_checkpoint_helper}" \
      --action "${action}" --pid "${pid}" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  local status=$?
  if (( status != 0 )); then
    cat "${controller_log}" >&2 || true
    cat "${images_dir}/dump.log" >&2 2>/dev/null || true
    cat "${images_dir}/restore.log" >&2 2>/dev/null || true
  fi
  for ((i=${#checkpointed[@]} - 1; i >= 0; --i)); do
    cleanup_action restore "${checkpointed[i]}"
  done
  for ((i=${#locked[@]} - 1; i >= 0; --i)); do
    cleanup_action unlock "${locked[i]}"
  done
  exit "${status}"
}
trap cleanup EXIT

mapfile -t process_tree_pids < <(
    "${python_bin}" "${host_tools}" process-tree \
      "${controller_pid}" "${run_dir}/worker-pids.json"
)

cuda_pids=()
for pid in "${process_tree_pids[@]}"; do
  if timeout 10s "${cuda_checkpoint_helper}" --get-restore-tid --pid "${pid}" \
      >/dev/null 2>&1; then
    cuda_pids+=("${pid}")
    pid_start_times["${pid}"]=$(process_start_time "${pid}")
  fi
done
if [[ ${#cuda_pids[@]} -lt 2 ]]; then
  echo "expected at least two CUDA processes, found ${#cuda_pids[@]}" >&2
  exit 1
fi
echo "CUDA checkpoint PIDs: ${cuda_pids[*]}"
if [[ -n "${source_gpu_uuids}" ]]; then
  wait_for_worker_gpu_residency before_checkpoint "${source_gpu_uuids}"
fi

cuda_migration_pids=()
if [[ -n "${cuda_device_map}" ]]; then
  if [[ -z "${source_gpu_uuids}" ]]; then
    echo "SGLANG_CRIU_SOURCE_GPU_UUIDS is required with SGLANG_CRIU_DEVICE_MAP" >&2
    exit 1
  fi
  mapfile -t cuda_migration_pids < <(
    "${python_bin}" "${host_tools}" migration-pids \
      "${source_gpu_uuids}" "${cuda_pids[@]}"
  )
  if [[ ${#cuda_migration_pids[@]} -eq 0 ]]; then
    echo "no CUDA checkpoint PID is resident on the source GPUs" >&2
    exit 1
  fi
fi
declare -A migrate_cuda_pid=()
for pid in "${cuda_migration_pids[@]}"; do
  migrate_cuda_pid["${pid}"]=1
done
echo "CUDA migration PIDs: ${cuda_migration_pids[*]:-none}"
echo "CUDA migration map: ${cuda_device_map:-none}"

for pid in "${cuda_pids[@]}"; do
  timeout 30s "${cuda_checkpoint_helper}" --action lock --pid "${pid}" --timeout 20000
  locked+=("${pid}")
done
for pid in "${cuda_pids[@]}"; do
  timeout 30s "${cuda_checkpoint_helper}" --action checkpoint --pid "${pid}"
  checkpointed+=("${pid}")
  state=$(timeout 10s "${cuda_checkpoint_helper}" --get-state --pid "${pid}")
  if [[ "${state}" != checkpointed ]]; then
    echo "CUDA process ${pid} is ${state}, expected checkpointed" >&2
    exit 1
  fi
done

timeout "${timeout_seconds}s" criu dump \
  --tree "${controller_pid}" \
  --images-dir "${images_dir}" \
  --shell-job --file-locks --link-remap --tcp-established --manage-cgroups=ignore \
  --ghost-limit 128M \
  --timeout "${criu_timeout_seconds}" \
  --libdir "${plugin_dir}" --log-file dump.log -v4

timeout "${timeout_seconds}s" criu restore \
  --images-dir "${images_dir}" \
  --shell-job --file-locks --link-remap --tcp-established --manage-cgroups=ignore \
  --timeout "${criu_timeout_seconds}" \
  --libdir "${plugin_dir}" --restore-detached \
  --log-file restore.log -v4

if grep -Eq "RESUME_DEVICES (RESTORE|UNLOCK) failed|Could not restore on process ID" "${images_dir}/restore.log"; then
  echo "CUDA process restore failed; refusing to resume the SGLang workload" >&2
  exit 1
fi

for pid in "${cuda_pids[@]}"; do
  pid_start_times["${pid}"]=$(process_start_time "${pid}")
done
for pid in "${cuda_pids[@]}"; do
  restore_args=(--action restore --pid "${pid}")
  if [[ -n "${migrate_cuda_pid[${pid}]:-}" ]]; then
    restore_args+=(--device-map "${cuda_device_map}")
  fi
  echo "Restoring CUDA PID ${pid} with device map: ${migrate_cuda_pid[${pid}]:-0}"
  timeout 30s "${cuda_checkpoint_helper}" "${restore_args[@]}"
done
checkpointed=()
for pid in "${cuda_pids[@]}"; do
  timeout 30s "${cuda_checkpoint_helper}" --action unlock --pid "${pid}"
done
locked=()
if [[ -n "${restore_gpu_uuids}" ]]; then
  wait_for_worker_gpu_residency after_restore "${restore_gpu_uuids}"
fi

printf 'restored\n' >"${run_dir}/phase.tmp"
mv "${run_dir}/phase.tmp" "${run_dir}/phase"
wait_for_file "${run_dir}/passed" "post-restore generation"
wait_for_process_exit "${controller_pid}" "${controller_start_time}"
launcher_status=0
wait "${launcher_pid}" || launcher_status=$?
# CRIU dump kills the original task tree. Bash retains that SIGKILL status even
# after the restored controller completes successfully under the same PID.
if [[ ${launcher_status} -ne 0 && ${launcher_status} -ne 137 ]]; then
  echo "cuda-checkpoint launcher exited with status ${launcher_status}" >&2
  exit 1
fi

cat "${controller_log}"
cat "${run_dir}/gsm8k-result.json"
echo "Qwen3 TP2 CUDA graph checkpoint/restore and GSM8K correctness passed"
