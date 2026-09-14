#!/bin/bash
#
# Full 1940-2025 evaluator rollout of the hybrid-residual arm-1 model.
# Default checkpoint: training_checkpoints/best_inference_ckpt.tar (epoch 15)
# from the arm-1 training results dataset. Override RESULTS_DATASET and/or
# CKPT_PATH to evaluate a different checkpoint.
#
# Examples:
#   ./launch-eval-1940-2025.sh
#   CKPT_PATH=training_checkpoints/ckpt.tar ./launch-eval-1940-2025.sh  # latest ckpt
#   DRY_RUN=1 ./launch-eval-1940-2025.sh

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
PRIORITY="${PRIORITY:-normal}"
JOB_GROUP="${JOB_GROUP:-era5-1deg-hybrid-residual}"
JOB_NAME="${JOB_NAME:-era5-1deg-hybridresid-ep15-eval-1940-2025}"
RESULTS_DATASET="${RESULTS_DATASET:-01M2DXYPGS2NJES6GAENPG84DD}"
CKPT_PATH="${CKPT_PATH:-training_checkpoints/best_inference_ckpt.tar}"
CONFIG_FILENAME="hybridresid-eval-1940-2025.yaml"

REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#"$REPO_ROOT"/}
CONFIG_PATH="${SCRIPT_PATH}/${CONFIG_FILENAME}"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}

cd "$REPO_ROOT"

python -m fme.ace.validate_config --config_type evaluator "$CONFIG_PATH"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "  [dry-run] $JOB_NAME ($CONFIG_PATH) ckpt ${RESULTS_DATASET}:${CKPT_PATH}"
  exit 0
fi

gantry run \
  --name "$JOB_NAME" \
  --task-name "$JOB_NAME" \
  --description "ERA5 1deg hybrid-residual full 1940-2025 evaluator rollout" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace \
  --priority "$PRIORITY" \
  --not-preemptible \
  --cluster ai2/ceres \
  --cluster ai2/jupiter \
  --cluster ai2/titan \
  --env WANDB_USERNAME="$WANDB_USERNAME" \
  --env WANDB_NAME="$JOB_NAME" \
  --env WANDB_JOB_TYPE=inference \
  --env WANDB_RUN_GROUP="$JOB_GROUP" \
  --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset-secret google-credentials:/tmp/google_application_credentials.json \
  --dataset "${RESULTS_DATASET}:${CKPT_PATH}:/ckpt.tar" \
  --gpus 1 \
  --shared-memory 50GiB \
  --weka climate-default:/climate-default \
  --budget ai2/atec-climate \
  --system-python \
  --install "pip install --no-deps ." \
  -- python -I -m fme.ace.evaluator "$CONFIG_PATH"
