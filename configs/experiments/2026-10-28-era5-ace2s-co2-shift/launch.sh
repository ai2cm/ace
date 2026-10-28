#!/bin/bash
# ACE-only CO2-clock test of the ERA5 4deg daily ACE2S: ARM=control|plus2yr|minus2yr ./launch.sh (default: all three).
# Job + wandb name era5-ace2s-4deg-co2shift-<ARM>, wandb group era5-ace2s-4deg-co2shift.
set -euo pipefail
ARMS="${ARM:-control plus2yr minus2yr}"
MIN_RUNTIME="${MIN_RUNTIME:-3h}"
PRIORITY="${PRIORITY:-high}"
CKPT_DS="${CKPT_DS:-01M39XBD8K0KB3GAQ79ETVN470}"   # 4deg-daily-ace2s-ft3-detached-rs0 (Jeremy, wandb 1r7oyu58)
CKPT_FILE="${CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
CO2_DS="${CO2_DS:-01M62CZ7J9ZXW810YFGD4GH8MN}"   # era5-4deg-daily-co2-shift-pm2yr (make_co2_shift_stores.py)
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
for A in $ARMS; do
  JOB="era5-ace2s-4deg-co2shift-${A}"
  out=$(gantry run --name "$JOB" --task-name "$JOB" \
    --description "ERA5 4deg daily ACE2S, ACE-only 1979-2023, CO2 arm ${A} (CO2-clock memorization test)" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --min-runtime "$MIN_RUNTIME" \
    --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
    --weka climate-default:/climate-default \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
    --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=era5-ace2s-4deg-co2shift \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset "${CKPT_DS}:${CKPT_FILE}:/ckpt.tar" \
    --dataset "${CO2_DS}:/co2shift" \
    --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- python -I -m fme.ace.evaluator "${SCRIPT_PATH}/evaluator-${A}.yaml" 2>&1) || true
  echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB: $(echo "$out" | grep -om1 'https://beaker.org/ex/[A-Z0-9]*')" || echo "FAILED $JOB: $(echo "$out" | tail -3)"
done
