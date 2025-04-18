#!/bin/bash
# uses the augusta cluster which doesn't have weka access but has GCS access and is
# typically more available than cirrascale clusters

set -e

# recommended but not required to change this
JOB_NAME="conus-downscaling-25km-to-3km-train-patch-fullsnapshot-emaval-v1"
CONFIG_FILENAME="train-conus-25km-to-3km-augusta.yaml"

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME

 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=8   # TODO: change to 8 after testing

cd $REPO_ROOT  # so config path is valid no matter where we are running this script
DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run \
    --name $JOB_NAME \
    --description 'Run downscaling 25km to 3km training over CONUS' \
    --workspace ai2/downscaling \
    --priority high \
    --preemptible \
    --cluster ai2/augusta-google-1 \
    --beaker-image $DEPS_ONLY_IMAGE \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP= \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --gpus $N_GPUS \
    --shared-memory 400GiB \
    --budget ai2/climate \
    --no-conda \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node $N_GPUS -m fme.downscaling.train $CONFIG_PATH
