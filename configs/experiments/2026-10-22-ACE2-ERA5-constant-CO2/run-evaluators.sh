#!/bin/bash
# Submit the standard and constant-CO2 ACE2-ERA5 evaluator jobs (1940 through 2020).
# Usage: bash configs/experiments/2026-10-22-ACE2-ERA5-constant-CO2/run-evaluators.sh [config_tag ...]
#   config_tag defaults to both: standard-1940-2020 constant-co2-1940-2020

set -e

JOB_GROUP="ace2-era5-constant-co2"
# ACE2-ERA5 checkpoint used in ai2cm/ace2-paper experiments/evaluator-era5-main/run.py
EXISTING_RESULTS_DATASET="01J4MT10JPQ8MFA41F2AXGFYJ9"
CONFIG_DIR=configs/experiments/2026-10-22-ACE2-ERA5-constant-CO2
TAGS=("$@")
if [[ ${#TAGS[@]} -eq 0 ]]; then
    TAGS=(standard-1940-2020 constant-co2-1940-2020)
fi

# since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=spencerc_ai2
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"  # so config paths are valid no matter where we are running this script

for TAG in "${TAGS[@]}"; do
    CONFIG_PATH="$CONFIG_DIR/evaluator-config-$TAG.yaml"
    JOB_NAME="ace2-era5-eval-$TAG"
    python -m fme.ace.validate_config --config_type evaluator "$CONFIG_PATH"
    gantry run \
        --name "$JOB_NAME" \
        --task-name "$JOB_NAME" \
        --description "Run ACE2-ERA5 evaluator: $TAG" \
        --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
        --workspace ai2/ace \
        --priority urgent \
        --min-runtime 3h \
        --cluster ai2/ceres \
        --cluster ai2/jupiter \
        --env WANDB_USERNAME="$BEAKER_USERNAME" \
        --env WANDB_NAME="$JOB_NAME" \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP="$JOB_GROUP" \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset "$EXISTING_RESULTS_DATASET:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar" \
        --gpus 1 \
        --shared-memory 50GiB \
        --weka climate-default:/climate-default \
        --system-python \
        --install "pip install --no-deps ." \
        -- python -I -m fme.ace.evaluator "$CONFIG_PATH"
done
