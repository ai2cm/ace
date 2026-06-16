#!/bin/bash
# Clean host-climate comparison for the labeled foundation model.
#
# Re-evaluates the labels that the original March evals ran against a
# non-comparable checkpoint, this time all from the SAME checkpoint: the
# all-models multi-step-FT best-inference snapshot (beaker
# 01KK5ZKY0P3QE36KHN63FQ4CNN), which is exactly the checkpoint the clean
# c96-shield eval (wandb ezybh824) already used. So era5 (was era5-only
# FT), x-shield (was pre-train only), and e3sm (was the latest, not
# best-inference, snapshot) are brought onto one checkpoint; the existing
# c96-shield eval completes the four-label set.
#
# Forked from exp/labeled-foundation-model @ ee2b78fb so Troy's branch is
# untouched. Evaluator configs are unchanged from that commit.
set -e

CKPT_DATASET="01KK5ZKY0P3QE36KHN63FQ4CNN"  # all-models FT best-inference
JOB_GROUP="label-foundation-clean-comparison"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_PATH="configs/experiments/2026-01-23-labeled-foundation-model"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')

# label -> evaluator config
declare -A CONFIGS=(
  [era5]="ace-evaluator-config-era5.yaml"
  [x-shield]="ace-evaluator-config-x-shield.yaml"
  [e3sm]="ace-evaluator-config-e3sm.yaml"
)

cd "$REPO_ROOT"
for LABEL in era5 x-shield e3sm; do
  CONFIG_PATH="$SCRIPT_PATH/${CONFIGS[$LABEL]}"
  JOB_NAME="label-foundation-clean-eval-${LABEL}-allmodels-bestinf"
  echo "=== launching $JOB_NAME ($CONFIG_PATH) ==="
  gantry run \
    --name "$JOB_NAME" \
    --task-name "$JOB_NAME" \
    --description 'Clean host-climate comparison: labeled FM evaluator, all-models best-inference ckpt' \
    --beaker-image "$(cat "$REPO_ROOT"/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority high \
    --not-preemptible \
    --cluster ai2/saturn-cirrascale \
    --cluster ai2/ceres-cirrascale \
    --env WANDB_USERNAME="$BEAKER_USERNAME" \
    --env WANDB_NAME="$JOB_NAME" \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP="$JOB_GROUP" \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "${CKPT_DATASET}:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar" \
    --gpus 1 \
    --shared-memory 50GiB \
    --weka climate-default:/climate-default \
    --allow-dirty \
    --yes \
    --system-python \
    --install "pip install --no-deps ." \
    -- python -I -m fme.ace.evaluator "$CONFIG_PATH"
done
