#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export AVATAR_PORT="${AVATAR_PORT:-7860}"
export AVATAR_HOST="${AVATAR_HOST:-127.0.0.1}"
export AVATAR_RUNS_ROOT="${AVATAR_RUNS_ROOT:-${PROJECT_ROOT}/outputs/interactive_avatar_runs}"
export AVATAR_WORKER_QUEUE_DIR="${AVATAR_WORKER_QUEUE_DIR:-${PROJECT_ROOT}/outputs/interactive_avatar_worker_queue}"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" -m uvicorn apps.interactive_avatar.server:app \
  --host "${AVATAR_HOST}" \
  --port "${AVATAR_PORT}"
