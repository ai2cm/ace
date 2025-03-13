#!/bin/bash

set -e

JOB_NAME="ft-ocn_only_loss-ocn_frac-grad_acc-10ep-train"  # recommnended but not required to change this
JOB_GROUP="ft-ocn_only_loss-ocn_frac-grad_acc-10ep"

EXISTING_RESULTS_ATMOS_DATASET="01JE8017VZVRBGCEK5S3DA5G08"  # this contains the pretrained atmosphere checkpoint
EXISTING_RESULTS_OCEAN_DATASET="01JNPV22RQF89C9WWG7J21JZD7"

CONFIG_FILENAME="coupled-train-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
# since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=8

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

# TODO: add fme.coupled.validate_config
# python -m fme.coupled.validate_config --config_type train $CONFIG_PATH

gantry run \
    --name $JOB_NAME \
    --description 'Run ACE coupled training' \
    --beaker-image oliverwm/fme-deps-only-5493f777 \
    --workspace ai2/ace \
    --priority high \
    --preemptible \
    --cluster ai2/jupiter-cirrascale-2 \
    --cluster ai2/ceres-cirrascale \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP=$JOB_GROUP \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset jamesd/2025-01-20-cm4-piControl-200yr-dataset-stats:/statsdata \
    --dataset $EXISTING_RESULTS_ATMOS_DATASET:training_checkpoints/best_inference_ckpt.tar:/atmos_ckpt.tar \
    --dataset $EXISTING_RESULTS_OCEAN_DATASET:training_checkpoints/best_inference_ckpt.tar:/ocean_ckpt.tar \
    --gpus $N_GPUS \
    --shared-memory 400GiB \
    --weka climate-default:/climate-default \
    --budget ai2/climate \
    --no-conda \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node $N_GPUS -m fme.coupled.train $CONFIG_PATH
