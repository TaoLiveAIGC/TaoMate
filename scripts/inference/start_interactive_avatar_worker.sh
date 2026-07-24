#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export AVATAR_WORKER_QUEUE_DIR="${AVATAR_WORKER_QUEUE_DIR:-${PROJECT_ROOT}/outputs/interactive_avatar_worker_queue}"
export AVATAR_WORKER_CUDA_VISIBLE_DEVICES="${AVATAR_WORKER_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export AVATAR_MASTER_PORT="${AVATAR_MASTER_PORT:-29731}"
export AVATAR_WORKER_POLL_INTERVAL="${AVATAR_WORKER_POLL_INTERVAL:-0.05}"
PYTHON_BIN="${PYTHON_BIN:-python}"

IFS=',' read -r -a WORKER_GPU_IDS <<< "${AVATAR_WORKER_CUDA_VISIBLE_DEVICES}"
if [ "${#WORKER_GPU_IDS[@]}" -ne 4 ]; then
  echo "AVATAR_WORKER_CUDA_VISIBLE_DEVICES must contain exactly four GPU IDs." >&2
  exit 2
fi
for GPU_ID in "${WORKER_GPU_IDS[@]}"; do
  if [ -z "${GPU_ID}" ]; then
    echo "AVATAR_WORKER_CUDA_VISIBLE_DEVICES contains an empty GPU ID." >&2
    exit 2
  fi
done

mkdir -p "${AVATAR_WORKER_QUEUE_DIR}"/{pending,running,done,failed}
rm -f "${AVATAR_WORKER_QUEUE_DIR}/worker_heartbeat.json"

exec env CUDA_VISIBLE_DEVICES="${AVATAR_WORKER_CUDA_VISIBLE_DEVICES}" \
  "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port="${AVATAR_MASTER_PORT}" \
    apps/interactive_avatar/worker.py \
    --queue_dir "${AVATAR_WORKER_QUEUE_DIR}" \
    --poll_interval "${AVATAR_WORKER_POLL_INTERVAL}"
