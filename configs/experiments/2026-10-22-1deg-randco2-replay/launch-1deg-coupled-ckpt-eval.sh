#!/bin/bash
# 200-yr piControl coupled rollout of one 1deg coupled checkpoint (a Beaker dataset holding the .tar).
#   JOB=<name> CKPT_DS=<dataset> CKPT_FILE=ckpt.tar ./launch-1deg-coupled-ckpt-eval.sh
set -euo pipefail
JOB="${JOB:?job name}"
CKPT_DS="${CKPT_DS:?checkpoint dataset}"
CKPT_FILE="${CKPT_FILE:-ckpt.tar}"
CFG="${CFG:-coupled-evaluator-config-piC-ic0151-200yr-1deg-singleckpt.yaml}"
MIN_RUNTIME="${MIN_RUNTIME:-6h}"   # gantry --min-runtime; 1deg template default
PRIORITY="${PRIORITY:-high}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "1deg coupled checkpoint ${CKPT_DS}: ${CFG}" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority "$PRIORITY" --min-runtime "$MIN_RUNTIME" \
  --cluster ai2/titan --cluster ai2/jupiter --cluster ai2/ceres \
  --weka climate-default:/climate-default \
  --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-1deg-centennial \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset "${CKPT_DS}:${CKPT_FILE}:/ckpt.tar" \
  --gpus 1 --shared-memory 200GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- python -I -m fme.coupled.evaluator "${SCRIPT_PATH}/eval/${CFG}" 2>&1) || true
echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB: $(echo "$out" | grep -om1 'https://beaker.org/ex/[A-Z0-9]*')" || { echo "FAILED $JOB: $(echo "$out" | tail -3)"; exit 1; }
