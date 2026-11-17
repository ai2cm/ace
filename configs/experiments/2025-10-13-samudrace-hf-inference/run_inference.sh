#!/bin/bash

set -e

JOB_NAME="v2025-06-03-inference-no-target-50yr-timing"
JOB_GROUP="v2025-06-03-inference-no-target-50yr-timing"
CONFIG_FILENAME="inference_config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH="${SCRIPT_PATH}${CONFIG_FILENAME}"
 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

python -m fme.coupled.validate_config --config_type inference $CONFIG_PATH

gantry run \
    --name $JOB_NAME \
    --task-name $JOB_NAME \
    --description 'Run ACE-Samudra inference no target' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority normal \
    --not-preemptible \
    --cluster ai2/ceres \
    --env WANDB_USERNAME=$WANDB_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP=$JOB_GROUP \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --gpus 1 \
    --shared-memory 50GiB \
    --weka climate-default:/climate-default \
    --budget ai2/climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- python -I -m fme.coupled.inference $CONFIG_PATH
