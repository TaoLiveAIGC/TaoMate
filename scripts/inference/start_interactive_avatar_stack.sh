#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

AVATAR_HOST="${AVATAR_HOST:-127.0.0.1}"
AVATAR_PORT="${AVATAR_PORT:-7860}"
AVATAR_LOG_DIR="${AVATAR_LOG_DIR:-${PROJECT_ROOT}/outputs/service_logs}"
AVATAR_RUNS_ROOT="${AVATAR_RUNS_ROOT:-${PROJECT_ROOT}/outputs/interactive_avatar_runs}"
AVATAR_WORKER_QUEUE_DIR="${AVATAR_WORKER_QUEUE_DIR:-${PROJECT_ROOT}/outputs/interactive_avatar_worker_queue}"
AVATAR_WORKER_CUDA_VISIBLE_DEVICES="${AVATAR_WORKER_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
AVATAR_MASTER_PORT="${AVATAR_MASTER_PORT:-29731}"
AVATAR_STARTUP_TIMEOUT="${AVATAR_STARTUP_TIMEOUT:-3600}"
DIALOGUE_HOST="${DIALOGUE_HOST:-127.0.0.1}"
DIALOGUE_PORT="${DIALOGUE_PORT:-7864}"
DIALOGUE_MODEL_NAME="${DIALOGUE_MODEL_NAME:-taolive-dialogue}"
DIALOGUE_STARTUP_TIMEOUT="${DIALOGUE_STARTUP_TIMEOUT:-600}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_CKPT="${MODEL_CKPT:-}"
BASE_MODEL_CKPT="${BASE_MODEL_CKPT:-}"
GEMMA_PATH="${GEMMA_PATH:-}"
DIALOGUE_MODEL_PATH="${DIALOGUE_MODEL_PATH:-${PROJECT_ROOT}/models/dialogue/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf}"
PROJECT_LLAMA_SERVER_BIN="${PROJECT_ROOT}/third_party/llama.cpp/build/bin/llama-server"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${PROJECT_LLAMA_SERVER_BIN}}"
ASR_MODEL_PATH="${ASR_MODEL_PATH:-${PROJECT_ROOT}/models/whisper/tiny.pt}"

WORKER_SESSION="taomate_worker"
SERVICE_SESSION="taomate_service"
DIALOGUE_SESSION="taomate_dialogue"

for path in "${MODEL_CKPT}" "${BASE_MODEL_CKPT}" "${DIALOGUE_MODEL_PATH}" "${ASR_MODEL_PATH}"; do
  if [ ! -f "${path}" ]; then
    echo "Required model file not found: ${path:-<unset>}" >&2
    exit 2
  fi
done
if [ ! -d "${GEMMA_PATH}" ]; then
  echo "GEMMA_PATH must point to the local text-encoder directory." >&2
  exit 2
fi
for command in tmux curl; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "${command} is required to start the demo stack." >&2
    exit 2
  fi
done
if ! "${PYTHON_BIN}" -c "import whisper" >/dev/null 2>&1; then
  echo "Python package openai-whisper is required. Run: ${PYTHON_BIN} -m pip install -r requirements.txt" >&2
  exit 2
fi

IFS=',' read -r -a WORKER_GPU_IDS <<< "${AVATAR_WORKER_CUDA_VISIBLE_DEVICES}"
if [ "${#WORKER_GPU_IDS[@]}" -ne 4 ]; then
  echo "AVATAR_WORKER_CUDA_VISIBLE_DEVICES must contain exactly four GPU IDs." >&2
  exit 2
fi
for gpu_id in "${WORKER_GPU_IDS[@]}"; do
  if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU ID in AVATAR_WORKER_CUDA_VISIBLE_DEVICES: ${gpu_id}" >&2
    exit 2
  fi
done
if [[ ! "${AVATAR_STARTUP_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "AVATAR_STARTUP_TIMEOUT must be a positive integer." >&2
  exit 2
fi
if [[ ! "${DIALOGUE_STARTUP_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "DIALOGUE_STARTUP_TIMEOUT must be a positive integer." >&2
  exit 2
fi

DIALOGUE_GPU="${DIALOGUE_GPU:-${WORKER_GPU_IDS[3]}}"
ASR_DEVICE="${ASR_DEVICE:-cuda:${DIALOGUE_GPU}}"
DIALOGUE_API_BASE="http://${DIALOGUE_HOST}:${DIALOGUE_PORT}/v1"
ACCESS_HOST="${AVATAR_HOST}"
if [ "${ACCESS_HOST}" = "0.0.0.0" ]; then
  ACCESS_HOST="127.0.0.1"
fi
DEMO_URL="http://${ACCESS_HOST}:${AVATAR_PORT}"
HEARTBEAT_PATH="${AVATAR_WORKER_QUEUE_DIR}/worker_heartbeat.json"

resolve_llama_server_bin() {
  local configured_bin="$1"
  local compute_cap=""
  local binary_sm=""
  local target_sm=""
  local sibling_bin=""

  if [ ! -x "${configured_bin}" ]; then
    echo "LLAMA_SERVER_BIN must point to an executable llama-server: ${configured_bin}" >&2
    return 2
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' "${configured_bin}"
    return 0
  fi

  compute_cap="$(nvidia-smi -i "${DIALOGUE_GPU}" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)"
  compute_cap="${compute_cap%%$'\n'*}"
  compute_cap="${compute_cap//[[:space:]]/}"
  target_sm="${compute_cap//./}"
  if [[ -z "${target_sm}" || ! "${configured_bin}" =~ sm([0-9]+) ]]; then
    printf '%s\n' "${configured_bin}"
    return 0
  fi

  binary_sm="${BASH_REMATCH[1]}"
  if [ "${binary_sm}" = "${target_sm}" ]; then
    printf '%s\n' "${configured_bin}"
    return 0
  fi

  sibling_bin="${configured_bin/sm${binary_sm}/sm${target_sm}}"
  if [ -x "${sibling_bin}" ]; then
    echo "llama-server targets sm${binary_sm}, but dialogue GPU ${DIALOGUE_GPU} is sm${target_sm}; using ${sibling_bin}." >&2
    printf '%s\n' "${sibling_bin}"
    return 0
  fi
  if [ -x "${PROJECT_LLAMA_SERVER_BIN}" ] && [ "${PROJECT_LLAMA_SERVER_BIN}" != "${configured_bin}" ]; then
    echo "llama-server targets sm${binary_sm}, but dialogue GPU ${DIALOGUE_GPU} is sm${target_sm}; using the project-local build." >&2
    printf '%s\n' "${PROJECT_LLAMA_SERVER_BIN}"
    return 0
  fi

  echo "LLAMA_SERVER_BIN targets sm${binary_sm}, but dialogue GPU ${DIALOGUE_GPU} requires sm${target_sm}. Build llama.cpp on this machine or provide a compatible binary." >&2
  return 2
}

LLAMA_SERVER_BIN="$(resolve_llama_server_bin "${LLAMA_SERVER_BIN}")"

mkdir -p "${AVATAR_LOG_DIR}" "${AVATAR_RUNS_ROOT}" "${AVATAR_WORKER_QUEUE_DIR}"/{pending,running,done,failed}
rm -f "${HEARTBEAT_PATH}"

for session in "${SERVICE_SESSION}" "${WORKER_SESSION}" "${DIALOGUE_SESSION}"; do
  tmux kill-session -t "${session}" 2>/dev/null || true
done

start_dialogue_session() {
  tmux kill-session -t "${DIALOGUE_SESSION}" 2>/dev/null || true
  tmux new-session -d -s "${DIALOGUE_SESSION}" \
    "cd '${PROJECT_ROOT}' && env DIALOGUE_HOST='${DIALOGUE_HOST}' DIALOGUE_PORT='${DIALOGUE_PORT}' DIALOGUE_MODEL_NAME='${DIALOGUE_MODEL_NAME}' DIALOGUE_GPU='${DIALOGUE_GPU}' DIALOGUE_MODEL_PATH='${DIALOGUE_MODEL_PATH}' LLAMA_SERVER_BIN='${LLAMA_SERVER_BIN}' bash scripts/inference/start_dialogue_server.sh 2>&1 | tee -a '${AVATAR_LOG_DIR}/dialogue.log'"
}

wait_for_dialogue() {
  local started_at="$(date +%s)"
  local now=""
  local elapsed=""
  while true; do
    if curl -fsS --max-time 2 "${DIALOGUE_API_BASE}/models" >/dev/null 2>&1; then
      return 0
    fi
    if ! tmux has-session -t "${DIALOGUE_SESSION}" 2>/dev/null; then
      return 1
    fi
    now="$(date +%s)"
    elapsed="$((now - started_at))"
    printf '[%3ss] Dialogue: loading\n' "${elapsed}"
    if [ "${elapsed}" -ge "${DIALOGUE_STARTUP_TIMEOUT}" ]; then
      return 1
    fi
    sleep 30
  done
}

print_stop_commands() {
  echo "  bash scripts/inference/stop_interactive_avatar_stack.sh"
  echo "  # or: tmux kill-session -t ${SERVICE_SESSION}"
  echo "  #     tmux kill-session -t ${WORKER_SESSION}"
  echo "  #     tmux kill-session -t ${DIALOGUE_SESSION}"
}

show_failure_logs() {
  for name in dialogue worker service; do
    echo "${name^} log: ${AVATAR_LOG_DIR}/${name}.log" >&2
    tail -n 30 "${AVATAR_LOG_DIR}/${name}.log" 2>/dev/null >&2 || true
  done
}

on_interrupt() {
  echo
  echo "Startup wait interrupted. The tmux services are still running."
  echo "Stop them with:"
  print_stop_commands
  exit 130
}
trap on_interrupt INT TERM

echo "TaoMate interactive demo"
echo "[1/4] Starting the dialogue model on GPU ${DIALOGUE_GPU}"
start_dialogue_session
if ! wait_for_dialogue; then
  echo "Dialogue service failed before becoming ready." >&2
  show_failure_logs
  echo "No worker or web service was started." >&2
  exit 1
fi
echo "Dialogue model is ready."

echo "[2/4] Starting the four-GPU resident worker on ${AVATAR_WORKER_CUDA_VISIBLE_DEVICES}"
tmux new-session -d -s "${WORKER_SESSION}" \
  "cd '${PROJECT_ROOT}' && env PYTHON_BIN='${PYTHON_BIN}' AVATAR_WORKER_QUEUE_DIR='${AVATAR_WORKER_QUEUE_DIR}' AVATAR_WORKER_CUDA_VISIBLE_DEVICES='${AVATAR_WORKER_CUDA_VISIBLE_DEVICES}' AVATAR_MASTER_PORT='${AVATAR_MASTER_PORT}' bash scripts/inference/start_interactive_avatar_worker.sh 2>&1 | tee -a '${AVATAR_LOG_DIR}/worker.log'"

echo "[3/4] Starting the web service on ${AVATAR_HOST}:${AVATAR_PORT}"
tmux new-session -d -s "${SERVICE_SESSION}" \
  "cd '${PROJECT_ROOT}' && env PYTHON_BIN='${PYTHON_BIN}' MODEL_CKPT='${MODEL_CKPT}' BASE_MODEL_CKPT='${BASE_MODEL_CKPT}' GEMMA_PATH='${GEMMA_PATH}' ASR_MODEL_PATH='${ASR_MODEL_PATH}' ASR_DEVICE='${ASR_DEVICE}' DIALOGUE_API_BASE='${DIALOGUE_API_BASE}' DIALOGUE_MODEL_NAME='${DIALOGUE_MODEL_NAME}' AVATAR_HOST='${AVATAR_HOST}' AVATAR_PORT='${AVATAR_PORT}' AVATAR_RUNS_ROOT='${AVATAR_RUNS_ROOT}' AVATAR_WORKER_QUEUE_DIR='${AVATAR_WORKER_QUEUE_DIR}' bash scripts/inference/start_interactive_avatar_service.sh 2>&1 | tee -a '${AVATAR_LOG_DIR}/service.log'"

echo "[4/4] Waiting for the dialogue model, resident model, and website"
STARTED_AT="$(date +%s)"
while true; do
  NOW="$(date +%s)"
  ELAPSED="$((NOW - STARTED_AT))"

  for session in "${DIALOGUE_SESSION}" "${WORKER_SESSION}" "${SERVICE_SESSION}"; do
    if ! tmux has-session -t "${session}" 2>/dev/null; then
      echo "Startup failed: tmux session ${session} exited." >&2
      show_failure_logs
      echo "Stop remaining services with:" >&2
      print_stop_commands >&2
      exit 1
    fi
  done

  DIALOGUE_STATE="loading"
  if curl -fsS --max-time 2 "${DIALOGUE_API_BASE}/models" >/dev/null 2>&1; then
    DIALOGUE_STATE="ready"
  fi

  READY_JSON="$(curl -fsS --max-time 3 "${DEMO_URL}/readyz" 2>/dev/null || true)"
  WEB_STATE="starting"
  if [ -n "${READY_JSON}" ]; then
    WEB_STATE="online"
  fi

  WORKER_STATE="starting"
  MODEL_STATE="loading"
  if [ -f "${HEARTBEAT_PATH}" ]; then
    HEARTBEAT_STATE="$("${PYTHON_BIN}" -c 'import json, sys; data=json.load(open(sys.argv[1], encoding="utf-8")); residency=str(data.get("model_residency", "")); print("{}|{}".format(data.get("status", "starting"), "ready" if residency in {"resident_runtime_cache_loaded", "loaded", "warm"} else "loading"))' "${HEARTBEAT_PATH}" 2>/dev/null || true)"
    if [ -n "${HEARTBEAT_STATE}" ]; then
      WORKER_STATE="${HEARTBEAT_STATE%%|*}"
      MODEL_STATE="${HEARTBEAT_STATE#*|}"
    fi
  fi

  printf '[%3ss] Dialogue: %-7s Web: %-8s Worker: %-14s Model: %s\n' \
    "${ELAPSED}" "${DIALOGUE_STATE}" "${WEB_STATE}" "${WORKER_STATE}" "${MODEL_STATE}"

  if [ "${DIALOGUE_STATE}" = "ready" ] && \
     [[ "${READY_JSON}" == *'"ready":true'* ]] && \
     curl -fsS --max-time 2 "${DEMO_URL}/" >/dev/null 2>&1; then
    trap - INT TERM
    echo
    echo "TaoMate demo is ready."
    echo "Open: ${DEMO_URL}/"
    echo "Logs: ${AVATAR_LOG_DIR}"
    echo
    echo "Stop the demo with:"
    print_stop_commands
    exit 0
  fi

  if [ "${ELAPSED}" -ge "${AVATAR_STARTUP_TIMEOUT}" ]; then
    echo "Startup timed out after ${AVATAR_STARTUP_TIMEOUT}s." >&2
    show_failure_logs
    echo "The tmux services are still running. Stop them with:" >&2
    print_stop_commands >&2
    exit 1
  fi
  sleep 30
done
