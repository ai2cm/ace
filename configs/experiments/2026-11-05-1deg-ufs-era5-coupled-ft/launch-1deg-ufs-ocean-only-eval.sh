#!/bin/bash
# Ocean-only full-record rollout (replay fluxes) of a 1deg UFS ocean checkpoint, 1 GPU (coupled cooling deep dive, 2026-11-10).
#   CKPT_DS=<dataset> [CKPT_FILE=training_checkpoints/best_inference_ckpt.tar] NAME=<label> ./launch-1deg-ufs-ocean-only-eval.sh
set -euo pipefail
CKPT_DS="${CKPT_DS:?dataset holding the ocean checkpoint}"
CKPT_FILE="${CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
NAME="${NAME:?label}"
CFG="${CFG:-ocean-only-evaluator-config-ufs1deg-30yr-1994.yaml}"
MIN_RUNTIME="${MIN_RUNTIME:-3h}"
PRIORITY="${PRIORITY:-high}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
JOB="samudra-ufs1deg-${NAME}-oceanonly-30yr-1994"
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "UFS 1deg ocean-only full-record rollout (replay fluxes) of ${NAME} (${CKPT_DS}:${CKPT_FILE})" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority "$PRIORITY" --min-runtime "$MIN_RUNTIME" \
  --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
  --weka climate-default:/climate-default \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-ufs1deg-oceanonly-eval \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset "${CKPT_DS}:${CKPT_FILE}:/ckpt.tar" \
  --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- python -I -m fme.ace.evaluator "${SCRIPT_PATH}/eval/${CFG}" 2>&1) || true
echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB: $(echo "$out" | grep -om1 'https://beaker.org/ex/[A-Z0-9]*')" || { echo "FAILED $JOB: $(echo "$out" | tail -3)"; exit 1; }
