#!/bin/bash

set -e

JOB_NAME="ptd_fixed-zero_fill-inst_norm--ace-samudra-cm4-9fb4l2ax--evaluator"  # recommnended but not required to change this
JOB_GROUP="ptd_fixed-zero_fill-inst_norm--ace-samudra-cm4-9fb4l2ax"

EXISTING_RESULTS_ATMOS_DATASET="01JE8017VZVRBGCEK5S3DA5G08"  # this contains the atmosphere checkpoint to use for inference
EXISTING_RESULTS_OCEAN_DATASET="01JNJVMWQ6KQ42BQDV9SCK7BQE"  # this contains the ocean checkpoint to use for inference

CONFIG_FILENAME="coupled-evaluator-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

# TODO: add fme.coupled.validate_config
# python -m fme.coupled.validate_config --config_type evaluator $CONFIG_PATH

cd $REPO_ROOT && gantry run \
    --name $JOB_NAME \
    --description 'Run ACE coupled evaluator' \
    --beaker-image oliverwm/fme-deps-only-5493f777 \
    --workspace ai2/ace \
    --priority high \
    --cluster ai2/jupiter-cirrascale-2 \
    --cluster ai2/ceres-cirrascale \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP=$JOB_GROUP \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset $EXISTING_RESULTS_ATMOS_DATASET:training_checkpoints/best_inference_ckpt.tar:/atmos_ckpt.tar \
    --dataset $EXISTING_RESULTS_OCEAN_DATASET:training_checkpoints/best_inference_ckpt.tar:/ocean_ckpt.tar \
    --gpus 1 \
    --shared-memory 20GiB \
    --weka climate-default:/climate-default \
    --budget ai2/climate \
    --no-conda \
    --install "pip install --no-deps ." \
    -- python -I -m fme.coupled.evaluator $CONFIG_PATH
