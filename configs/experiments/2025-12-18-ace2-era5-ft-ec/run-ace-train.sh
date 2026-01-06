#!/bin/bash

set -e

JOB_NAME="ace-aimip-train-rs3-1s-b1-ft-ec-detect-anomaly-debug"
JOB_GROUP="ace21-era5"
CONFIG_FILENAME="ace-train-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH="${SCRIPT_PATH}${CONFIG_FILENAME}"
WANDB_USERNAME=spencerc_ai2
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=1

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

PRE_TRAINED_CHECKPOINT="01K9B1MXD6V26S8BQH5CKY514C"
CHECKPOINT_PATH=training_checkpoints/best_inference_ckpt.tar
override="max_epochs=1 n_forward_steps=1 stepper.parameter_init.weights_path=/pre-trained-checkpoint/ckpt.tar"
python -m fme.ace.validate_config --config_type train $CONFIG_PATH --override $override

gantry run \
    --name $JOB_NAME \
    --task-name $JOB_NAME \
    --description 'Run ACE training on AIMIP data' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/climate-titan \
    --priority urgent \
    --preemptible \
    --cluster ai2/titan \
    --env WANDB_USERNAME=$WANDB_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP=$JOB_GROUP \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset oliverwm/era5-1deg-8layer-stats-1990-2019-v2:/statsdata \
    --dataset $PRE_TRAINED_CHECKPOINT:$CHECKPOINT_PATH:/pre-trained-checkpoint/ckpt.tar \
    --gpus $N_GPUS \
    --shared-memory 400GiB \
    --weka climate-default:/climate-default \
    --budget ai2/climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_PATH --override $override
