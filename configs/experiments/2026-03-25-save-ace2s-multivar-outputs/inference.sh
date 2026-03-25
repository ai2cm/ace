#!/bin/bash

set -e

JOB_NAME="ace2s-evaluator-multivar-outputs-10yr-10ic"
JOB_GROUP=""
EXISTING_RESULTS_DATASET="01K9B2ZY6E1W5N34HQCVMS7BJM"  # this contains the checkpoint to use for inference
CONFIG_FILENAME="config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

STATS_DATASET="01K5A6EH0XE13D7RYWW4GGWCNE"

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

python -m fme.ace.validate_config --config_type evaluator $CONFIG_PATH

cd $REPO_ROOT && gantry run \
    --name $JOB_NAME \
    --task-name $JOB_NAME \
    --description 'Run ACE2S evaluator' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/climate-titan \
    --priority high \
    --not-preemptible \
    --cluster ai2/saturn \
    --cluster ai2/jupiter \
    --cluster ai2/ceres  \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP=$JOB_GROUP \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-annak \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset $EXISTING_RESULTS_DATASET:/weights \
    --dataset $STATS_DATASET:/statsdata \
    --gpus 1 \
    --shared-memory 50GiB \
    --weka climate-default:/climate-default \
    --budget ai2/climate \
    --install "pip install --no-deps ." \
    --allow-dirty \
    -- python -I -m fme.ace.evaluator $CONFIG_PATH
