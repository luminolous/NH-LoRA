#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/run_custom.yaml"
BENCHMARK="custom"
SEEDS=(1 2 3 4 5)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

LOG_DIR=$(python -c "from src.utils.config import load_config; print(load_config('${CONFIG}')['experiment']['log_dir'])")

mkdir -p "${LOG_DIR}"

echo "[INFO] Running ${BENCHMARK} with config ${CONFIG}"

for SEED in "${SEEDS[@]}"; do
  LOG_FILE="${LOG_DIR}/${BENCHMARK}_seed${SEED}_${TIMESTAMP}.log"
  echo "[INFO] Seed ${SEED} -> ${LOG_FILE}"

  PYTHONUNBUFFERED=1 NH_LORA_DISABLE_FILE_LOG=1 python -m src.engine.train \
    --config "${CONFIG}" \
    --seed "${SEED}" \
    --benchmark "${BENCHMARK}" \
    2>&1 | tee "${LOG_FILE}"
done

python -m src.engine.summarize \
  --config "${CONFIG}" \
  --benchmark "${BENCHMARK}"

echo "[INFO] Finished ${BENCHMARK}"
