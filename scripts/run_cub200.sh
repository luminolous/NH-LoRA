#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/cub200.yaml"
BENCHMARK="cub200"
SEEDS=(1 2 3 4 5)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p outputs/logs outputs/metrics/${BENCHMARK} outputs/summaries outputs/checkpoints/${BENCHMARK}

echo "[INFO] Running ${BENCHMARK} with config ${CONFIG}"

for SEED in "${SEEDS[@]}"; do
  LOG_FILE="outputs/logs/${BENCHMARK}_seed${SEED}_${TIMESTAMP}.log"
  echo "[INFO] Seed ${SEED} -> ${LOG_FILE}"

  python -m src.engine.train \
    --config "${CONFIG}" \
    --seed "${SEED}" \
    --benchmark "${BENCHMARK}" \
    --output-root outputs \
    >> "${LOG_FILE}" 2>&1

done

python -m src.engine.summarize \
  --metrics-dir "outputs/metrics/${BENCHMARK}" \
  --output-file "outputs/summaries/${BENCHMARK}_summary.json"

echo "[INFO] Finished ${BENCHMARK}"
