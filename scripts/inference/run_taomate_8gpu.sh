#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

BENCHMARK_JSON="${BENCHMARK_JSON:-configs/inference/benchmark_1min_windowed.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/taomate_8gpu}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29780}"

mkdir -p "${OUTPUT_ROOT}"

pids=()
for idx in 0 1 2 3; do
  gpu_a=$((idx * 2))
  gpu_b=$((gpu_a + 1))
  out_dir="${OUTPUT_ROOT}/shard_${idx}"
  GPU="${gpu_a},${gpu_b}" \
  MASTER_PORT="$((MASTER_PORT_BASE + idx))" \
  CASE_START="$((idx * 5))" \
  MAX_CASES=5 \
  BENCHMARK_JSON="${BENCHMARK_JSON}" \
  OUTPUT_DIR="${out_dir}" \
    bash scripts/inference/run_taomate_2gpu.sh &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "All shards finished: ${OUTPUT_ROOT}"
