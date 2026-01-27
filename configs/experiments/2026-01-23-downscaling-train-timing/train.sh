#!/bin/bash

set -e

#JOB_NAME="xshield-downscaling-apex-gnorm-channels-last-unetv2-bsz20-ch128"
#CONFIG_FILENAME="train_channels_last.yaml"
JOB_NAME="xshield-downscaling-torch-gnorm-unetv2-bsz16-ch128"
CONFIG_FILENAME="train_control.yaml"

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
wandb_group=""

BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=1 # TODO: change to 8 after testing

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

IMAGE=annak/deps-only-with-apex-v2  #$(cat $REPO_ROOT/latest_deps_only_image.txt)

gantry run \
    --name $JOB_NAME \
    --description 'Run downscaling 100km to 3km training global' \
    --workspace ai2/ace \
    --priority high \
    --preemptible \
    --cluster ai2/titan \
    --beaker-image $IMAGE \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP=$wandb_group \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --weka climate-default:/climate-default \
    --gpus $N_GPUS \
    --shared-memory 400GiB \
    --budget ai2/climate \
    --no-conda \
    --install "pip install --no-deps ." \
    --allow-dirty \
    -- torchrun --nproc_per_node $N_GPUS -m fme.downscaling.train $CONFIG_PATH