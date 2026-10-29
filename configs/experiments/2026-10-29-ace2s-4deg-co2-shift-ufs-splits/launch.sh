#!/bin/bash
# 4deg ERA5 ACE2S with the UFS SamudrACE split, with and without a random per-window CO2 time shift.
#   ARM=co2shift10 STAGE=pretrain ./launch.sh
#   ARM=co2shift10 STAGE=finetune PRETRAIN_DS=<pretrain results dataset> ./launch.sh
# One GPU per job, as the 4deg recipe (pretrain ~12.5 h, fine-tune ~6 h on one GPU).
set -euo pipefail
ARM="${ARM:?co2shift0 | co2shift10 | co2shift30}"
STAGE="${STAGE:?pretrain | finetune}"
MIN_RUNTIME="${MIN_RUNTIME:-8h}"
PRIORITY="${PRIORITY:-high}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WEIGHTS=()
if [ "$STAGE" = finetune ]; then
  WEIGHTS=(--dataset "${PRETRAIN_DS:?pretrain results dataset}:/weights")
  JOB="4deg-daily-ace2s-ufssplit-${ARM}-ft3-detached-rs0"
else
  JOB="4deg-daily-ace2s-ufssplit-${ARM}-pretrain-rs0"
fi
CFG="${SCRIPT_PATH}/${ARM}-${STAGE}.yaml"
cd "$REPO_ROOT"
python -m fme.ace.validate_config --config_type train "$CFG" > /dev/null
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "4deg ERA5 ACE2S, UFS split, ${ARM} ${STAGE}" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority "$PRIORITY" --min-runtime "$MIN_RUNTIME" \
  --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
  --weka climate-default:/climate-default \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=training --env WANDB_RUN_GROUP=era5-ace2s-4deg-co2shift-ufssplit \
  --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset-secret google-credentials:/tmp/google_application_credentials.json \
  "${WEIGHTS[@]}" \
  --gpus 1 --shared-memory 50GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- torchrun --nproc_per_node 1 -m fme.ace.train "$CFG" 2>&1) || true
echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB" || { echo "FAILED $JOB: $(echo "$out" | tail -3)"; exit 1; }
