#!/bin/bash

set -e

JOB_NAME="tend-pred-new-implementation-normed"  # recommended but not required to change this
CONFIG_PATH=configs/experiments/2025-03-17-ace2-som-residual-prediction/config.yaml  # relative to the root of the repository
# since we use a service account API key for wandb, we use the beaker username to set the wandb username
WANDB_USERNAME=oliverwm
WANDB_RUN_GROUP=ace2-som-tendency-prediction
WANDB_PROJECT=ace-som-mse
N_GPUS=8

python -m fme.ace.validate_config --config_type train $CONFIG_PATH

gantry run \
    --name $JOB_NAME \
    --description 'Try new residual prediction implementaiton' \
    --beaker-image oliverwm/fme-deps-only-2025-01-16 \
    --workspace ai2/ace \
    --priority high \
    --preemptible \
    --cluster ai2/jupiter-cirrascale-2 \
    --cluster ai2/ceres-cirrascale \
    --env WANDB_USERNAME=$WANDB_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_PROJECT=$WANDB_PROJECT \
    --env WANDB_RUN_GROUP=$WANDB_RUN_GROUP \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset annak/2024-09-13-vertically-resolved-1deg-c96-shield-som-increasing-co2-fme-stats:/statsdata \
    --gpus $N_GPUS \
    --shared-memory 400GiB \
    --weka climate-default:/climate-default \
    --budget ai2/climate \
    --no-conda \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_PATH
