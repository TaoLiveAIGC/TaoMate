#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
CASE_START="${CASE_START:-0}"
MAX_CASES="${MAX_CASES:-1}"
MODEL_CKPT="${MODEL_CKPT:-}"
BASE_MODEL_CKPT="${BASE_MODEL_CKPT:-}"
GEMMA_PATH="${GEMMA_PATH:-}"
BENCHMARK_JSON="${BENCHMARK_JSON:-configs/inference/benchmark_1min_windowed.json}"
PROMPT_CACHE="${PROMPT_CACHE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/taomate_1gpu}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-512}"
VIDEO_WIDTH="${VIDEO_WIDTH:-768}"
OUTPUT_VIDEO_HEIGHT="${OUTPUT_VIDEO_HEIGHT:-0}"
OUTPUT_VIDEO_WIDTH="${OUTPUT_VIDEO_WIDTH:-0}"

if [[ ! "${GPU}" =~ ^[0-9]+$ ]]; then
  echo "GPU must contain exactly one GPU ID." >&2
  exit 2
fi
if [ ! -f "${MODEL_CKPT}" ] || [ ! -f "${BASE_MODEL_CKPT}" ] || [ ! -d "${GEMMA_PATH}" ]; then
  echo "Set MODEL_CKPT, BASE_MODEL_CKPT, and GEMMA_PATH to valid local paths." >&2
  exit 2
fi
if [ ! -f "${BENCHMARK_JSON}" ]; then
  echo "Benchmark JSON not found: ${BENCHMARK_JSON}" >&2
  exit 2
fi

PROMPT_CACHE_ARGS=()
if [ -n "${PROMPT_CACHE}" ]; then
  PROMPT_CACHE_ARGS=(--prompt_cache_path "${PROMPT_CACHE}")
fi

mkdir -p "${OUTPUT_DIR}"

exec env \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  RANK=0 \
  LOCAL_RANK=0 \
  WORLD_SIZE=1 \
  PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" scripts/inference/infer_1min_windowed.py \
    --model_ckpt "${MODEL_CKPT}" \
    --base_model_ckpt "${BASE_MODEL_CKPT}" \
    --gemma_path "${GEMMA_PATH}" \
    --benchmark_json "${BENCHMARK_JSON}" \
    "${PROMPT_CACHE_ARGS[@]}" \
    --output_dir "${OUTPUT_DIR}" \
    --video_height "${VIDEO_HEIGHT}" \
    --video_width "${VIDEO_WIDTH}" \
    --output_video_height "${OUTPUT_VIDEO_HEIGHT}" \
    --output_video_width "${OUTPUT_VIDEO_WIDTH}" \
    --case_start "${CASE_START}" \
    --max_cases "${MAX_CASES}"
