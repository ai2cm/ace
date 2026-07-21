#!/bin/bash

set -euo pipefail

CONFIG_FILENAME="${CONFIG_FILENAME:-evaluate-nino-scalars.yaml}"
JOB_NAME="${JOB_NAME:-cm4-samudra-nino-oos-one-step}"
JOB_GROUP="${JOB_GROUP:-cm4-1pct-samudra-nino}"
OCEAN_RESULTS_DATASET="01KXKZ85HTDSGGXWD2DPW2QRFW"
OCEAN_CKPT="${OCEAN_CKPT:-best_inference_ckpt}"

REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
CONFIG_PATH="${SCRIPT_PATH}/${CONFIG_FILENAME}"
COMPACT_SCRIPT="${SCRIPT_PATH}/compact_nino_scalar_forecasts.py"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
N_GPUS=1

cd "$REPO_ROOT"

python -m fme.ace.validate_config --config_type evaluator "$CONFIG_PATH"

gantry run \
    --name "$JOB_NAME" \
    --task-name "$JOB_NAME" \
    --description "60 out-of-sample one-step Samudra Nino3.4 scalar forecasts" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/climate-titan \
    --priority urgent \
    --preemptible \
    --cluster ai2/titan \
    --weka climate-default:/climate-default \
    --env WANDB_USERNAME="$BEAKER_USERNAME" \
    --env WANDB_NAME="$JOB_NAME" \
    --env WANDB_JOB_TYPE=evaluation \
    --env WANDB_RUN_GROUP="$JOB_GROUP" \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "$OCEAN_RESULTS_DATASET:training_checkpoints/${OCEAN_CKPT}.tar:/ckpt.tar" \
    --gpus "$N_GPUS" \
    --shared-memory 400GiB \
    --budget ai2/atec-climate \
    --allow-dirty \
    --system-python \
    --install "pip install --no-deps ." \
    -- bash -c \
        "python -m fme.ace.evaluator '$CONFIG_PATH' && \
         python '$COMPACT_SCRIPT' --input-dir /results/raw --output-dir /results/nino_scalar_forecasts"
