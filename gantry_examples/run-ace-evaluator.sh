#!/bin/bash

set -e

JOB_NAME="ace-evaluator"  # recommnended but not required to change this
EXISTING_RESULTS_DATASET="01J4MT10JPQ8MFA41F2AXGFYJ9"  # this contains the checkpoint to use for inference
CONFIG_PATH=gantry_examples/ace-evaluator-config.yaml  # relative to the root of the repository
 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')

python -m fme.ace.validate_config --config_type evaluator $CONFIG_PATH

gantry run \
    --name $JOB_NAME \
    --description 'Run ACE evaluator' \
    --beaker-image elynn/fme-deps-only-2025-01 \
    --workspace ai2/ace \
    --priority normal \
    --cluster ai2/saturn-cirrascale \
    --cluster ai2/jupiter-cirrascale-2 \
    --cluster ai2/ceres-cirrascale \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP= \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset $EXISTING_RESULTS_DATASET:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar \
    --gpus 1 \
    --shared-memory 20GiB \
    --weka climate-default:/climate-default \
    --budget ai2/climate \
    --no-conda \
    --install "pip install --no-deps ./fme" \
    -- python -I -m fme.ace.evaluator $CONFIG_PATH
