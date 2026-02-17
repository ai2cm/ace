#!/bin/bash

set -e

JOB_NAME="BK-v1124-ACE2S-n512-e1c9-ft-20step-E3SMv3-piControl-eval-best-ckpt"
JOB_GROUP="BK-v1124-ACE2S-n512-e1c9-ft-20step-E3SMv3-piControl-100yr-eval"
EXISTING_RESULTS_DATASET="01KENDK3NS5SRPAHSTZEQXP4JF"  # this contains the checkpoint to use for inference
CONFIG_FILENAME="ace-evaluator-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH="${SCRIPT_PATH}${CONFIG_FILENAME}"
 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

python -m fme.ace.validate_config --config_type evaluator $CONFIG_PATH

gantry run \
    --name $JOB_NAME \
    --task-name $JOB_NAME \
    --description 'Run ACE evaluator for E3SMv3 atmosphere data' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority high \
    --not-preemptible \
    --cluster ai2/saturn \
    --cluster ai2/ceres \
    --cluster ai2/jupiter \
    --cluster ai2/neptune \
    --env WANDB_USERNAME=$WANDB_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP=$JOB_GROUP \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset $EXISTING_RESULTS_DATASET:training_checkpoints/best_ckpt.tar:/ckpt.tar \
    --gpus 1 \
    --shared-memory 50GiB \
    --weka climate-default:/climate-default \
    --budget ai2/climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- python -I -m fme.ace.evaluator $CONFIG_PATH
