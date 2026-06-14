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
mkdir -p "${state_root}"
state_dir=$(mktemp -d "${state_root%/}/run.XXXXXX")
run_dir="${state_dir}/run"
images_dir="${state_dir}/images"
plugin_dir="${state_dir}/plugins"
dist_store="${run_dir}/torch-dist-store"
mkdir -p "${run_dir}" "${images_dir}" "${plugin_dir}"

"${uv_bin}" pip install --system --break-system-packages --no-deps \
  --editable "${movin_root}"
MOVIN_DEPS_DIR="${movin_deps_dir}" "${python_bin}" -m movin.build_deps
"${python_bin}" -c 'import movin; print("movin package:", movin.__file__)'

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
      PYTHONPATH="${sglang_root}/python${PYTHONPATH:+:${PYTHONPATH}}" \
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

wait_for_file "${run_dir}/job-ready" "SGLang checkpoint barrier"
controller_pid=$(<"${run_dir}/controller-pid")

process_start_time() {
  python3 - "$1" <<'PY'
import sys
from pathlib import Path

stat = Path(f"/proc/{sys.argv[1]}/stat").read_text()
print(stat.rsplit(")", 1)[1].split()[19])
PY
}

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
    python3 - "${controller_pid}" "${run_dir}/worker-pids.json" <<'PY'
import json
import sys
from pathlib import Path

root = int(sys.argv[1])
workers = [int(pid) for pid in json.loads(Path(sys.argv[2]).read_text())]
children: dict[int, list[int]] = {}
for status_path in Path("/proc").glob("[0-9]*/status"):
    try:
        fields = {}
        for line in status_path.read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator:
                fields[key] = value.strip()
        pid = int(fields["Pid"])
        parent = int(fields["PPid"])
    except (OSError, KeyError, ValueError):
        continue
    children.setdefault(parent, []).append(pid)
for child_pids in children.values():
    child_pids.sort()

seen: set[int] = set()
def emit_subtree(pid: int) -> None:
    if pid in seen:
        return
    seen.add(pid)
    print(pid)
    for child in children.get(pid, ()):
        emit_subtree(child)

seen.add(root)
print(root)
for worker in workers:
    emit_subtree(worker)
for child in children.get(root, ()):
    emit_subtree(child)
PY
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
  timeout 30s "${cuda_checkpoint_helper}" --action restore --pid "${pid}"
done
checkpointed=()
for pid in "${cuda_pids[@]}"; do
  timeout 30s "${cuda_checkpoint_helper}" --action unlock --pid "${pid}"
done
locked=()

printf 'restored\n' >"${run_dir}/phase.tmp"
mv "${run_dir}/phase.tmp" "${run_dir}/phase"
wait_for_file "${run_dir}/passed" "post-restore generation"

cat "${controller_log}"
cat "${run_dir}/gsm8k-result.json"
echo "Qwen3 TP2 CUDA graph checkpoint/restore and GSM8K correctness passed"
