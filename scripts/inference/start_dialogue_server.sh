#!/usr/bin/env bash
set -euo pipefail

DIALOGUE_HOST="${DIALOGUE_HOST:-127.0.0.1}"
DIALOGUE_PORT="${DIALOGUE_PORT:-7864}"
DIALOGUE_MODEL_NAME="${DIALOGUE_MODEL_NAME:-taolive-dialogue}"
DIALOGUE_CONTEXT_SIZE="${DIALOGUE_CONTEXT_SIZE:-4096}"
DIALOGUE_GPU="${DIALOGUE_GPU:-3}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-}"
DIALOGUE_MODEL_PATH="${DIALOGUE_MODEL_PATH:-}"

if [ ! -x "${LLAMA_SERVER_BIN}" ]; then
  echo "LLAMA_SERVER_BIN must point to an executable llama-server." >&2
  exit 2
fi
if [ ! -f "${DIALOGUE_MODEL_PATH}" ]; then
  echo "DIALOGUE_MODEL_PATH must point to a GGUF dialogue model." >&2
  exit 2
fi

exec env CUDA_VISIBLE_DEVICES="${DIALOGUE_GPU}" "${LLAMA_SERVER_BIN}" \
  -m "${DIALOGUE_MODEL_PATH}" \
  --host "${DIALOGUE_HOST}" \
  --port "${DIALOGUE_PORT}" \
  --alias "${DIALOGUE_MODEL_NAME}" \
  --ctx-size "${DIALOGUE_CONTEXT_SIZE}" \
  --parallel 1 \
  --gpu-layers all \
  --main-gpu 0 \
  --split-mode none \
  --no-webui \
  --jinja \
  --reasoning off
