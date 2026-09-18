#!/bin/bash
# Evaluator run to see how the reworked Nino3.4 power spectrum plot (PR #1494)
# looks on a real model, using the ocean_wide hybridresid checkpoint.

set -e

JOB_NAME="enso-spectrum-pr1494-piC-eval"
JOB_GROUP="cm4_1pct_46to125_piC_156to235-ocean_wide_hybridresid_clip1_sstff_ohc"
# result dataset of elynn/...-ocean_wide_hybridresid_clip1_sstff_ohc-rs0-train
EXISTING_RESULTS_DATASET="01M2PEDZQZN2QJ7PVEF391X8T2"
CONFIG_FILENAME="evaluator-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH="${SCRIPT_PATH}${CONFIG_FILENAME}"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

python -m fme.ace.validate_config --config_type evaluator $CONFIG_PATH

gantry run \
    --name $JOB_NAME \
    --task-name $JOB_NAME \
    --description "ENSO spectrum plot check (PR #1494), ocean_wide hybridresid ckpt, piControl 20yr x 8 ICs" \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority normal \
    --not-preemptible \
    --cluster ai2/ceres \
    --cluster ai2/saturn \
    --cluster ai2/neptune \
    --weka climate-default:/climate-default \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
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
