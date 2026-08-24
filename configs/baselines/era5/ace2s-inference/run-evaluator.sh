#!/bin/bash
# Launch ACE2S-ERA5 inference evaluation runs.
# Usage: ./run-evaluator.sh [filter]
# e.g. ./run-evaluator.sh 10yr-IC0  # launch only 10yr IC0

set -e

CKPT_DATASET="01M0DVJG77ND6N2ENP8CPZ758Z"
JOB_GROUP="ace2s-era5-inference"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_PATH=$(git rev-parse --show-prefix)
FILTER="${1:-}"

run_inference() {
    local CONFIG_FILENAME=$1
    local JOB_NAME=$2
    local GPU_COUNT=${3:-1}

    if [[ -n "$FILTER" && "$JOB_NAME" != *"$FILTER"* ]]; then
        return
    fi

    CONFIG_PATH="${SCRIPT_PATH}${CONFIG_FILENAME}"

    echo "Launching: $JOB_NAME"
    echo "  Config: $CONFIG_PATH"
    echo "  Checkpoint: $CKPT_DATASET"

    cd "$REPO_ROOT"

    python -m fme.ace.validate_config --config_type evaluator "$CONFIG_PATH"

    gantry run \
        --name "$JOB_NAME" \
        --task-name "$JOB_NAME" \
        --description "ACE2S-ERA5 inference: $JOB_NAME" \
        --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
        --workspace ai2/ace \
        --priority high \
        --cluster ai2/jupiter-cirrascale-2 \
        --cluster ai2/saturn-cirrascale \
        --env WANDB_USERNAME=mcgibbon \
        --env WANDB_NAME="$JOB_NAME" \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP="$JOB_GROUP" \
        --env CM_PRIORITY=high \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset "$CKPT_DATASET":training_checkpoints/best_inference_ckpt.tar:/ckpt.tar \
        --gpus "$GPU_COUNT" \
        --shared-memory 50GiB \
        --weka climate-default:/climate-default \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- python -I -m fme.ace.evaluator "$CONFIG_PATH"
}

# 10yr runs (3 ICs) — save full predictions for tropical/strat analysis
run_inference "evaluator-10yr-IC0.yaml" "ace2s-era5-10yr-IC0"
run_inference "evaluator-10yr-IC1.yaml" "ace2s-era5-10yr-IC1"
run_inference "evaluator-10yr-IC2.yaml" "ace2s-era5-10yr-IC2"

# 81yr runs (3 ICs)
run_inference "evaluator-81yr-IC0.yaml" "ace2s-era5-81yr-IC0"
run_inference "evaluator-81yr-IC1.yaml" "ace2s-era5-81yr-IC1"
run_inference "evaluator-81yr-IC2.yaml" "ace2s-era5-81yr-IC2"

# Weather forecast (15-day from 2020)
run_inference "evaluator-weather-2020.yaml" "ace2s-era5-weather-2020"
