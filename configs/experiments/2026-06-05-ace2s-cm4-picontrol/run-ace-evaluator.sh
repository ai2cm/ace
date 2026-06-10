#!/bin/bash

set -e

JOB_GROUP="ace2s-cm4-picontrol"
# Beaker dataset IDs containing training checkpoints, one per random seed
EXISTING_RESULTS_DATASETS=(
    ""  # rs0
    ""  # rs1
)
CONFIG_FILENAME="ace-evaluator-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH="${SCRIPT_PATH}${CONFIG_FILENAME}"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-bhenn1983}
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

python -m fme.ace.validate_config --config_type evaluator $CONFIG_PATH

for RS in "${!EXISTING_RESULTS_DATASETS[@]}"; do
    EXISTING_RESULTS_DATASET="${EXISTING_RESULTS_DATASETS[$RS]}"
    JOB_NAME="ace2s-cm4-picontrol-evaluator-rs${RS}"

    gantry run \
        --name $JOB_NAME \
        --task-name $JOB_NAME \
        --description 'Run ACE2S CM4 piControl atmosphere evaluator' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority normal \
        --not-preemptible \
        --cluster ai2/ceres-cirrascale \
        --cluster ai2/jupiter-cirrascale \
        --cluster ai2/saturn-cirrascale \
        --weka climate-default:/climate-default \
        --env WANDB_USERNAME=$WANDB_USERNAME \
        --env WANDB_NAME=$JOB_NAME \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP=$JOB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $EXISTING_RESULTS_DATASET:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar \
        --gpus 1 \
        --shared-memory 50GiB \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- python -I -m fme.ace.evaluator $CONFIG_PATH
done
