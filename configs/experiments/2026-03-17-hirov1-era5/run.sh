#!/bin/bash

set -e

JOB_NAME="predict-HiROv1-on-era5-100km-to-3km-conus"

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')
CONFIG_PATH="$SCRIPT_PATH/config-plus4k-tropic.yaml"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

NGPU=8
IMAGE="$(cat $REPO_ROOT/latest_deps_only_image.txt)"

EXISTING_RESULTS_DATASET=01K8RWE83W8BEEAT2KRS94FVCD # best hist checkpoint from job using global validation

wandb_group=""

gantry run \
    --name $JOB_NAME \
    --description 'Run 100km to 3km predict on 1deg ERA5' \
    --workspace ai2/climate-titan \
    --priority urgent \
    --preemptible \
    --cluster ai2/titan \
    --beaker-image $IMAGE \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP=$wandb_group \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset $EXISTING_RESULTS_DATASET:checkpoints:/checkpoints \
    --weka climate-default:/climate-default \
    --gpus $NGPU \
    --shared-memory 400GiB \
    --budget ai2/climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node $NGPU -m fme.downscaling.predict $CONFIG_PATH
