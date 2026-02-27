#!/bin/bash
# uses the augusta cluster which doesn't have weka access but has GCS access and is
# typically more available than cirrascale clusters

set -e

# recommended but not required to change this

JOB_NAME="xshield-downscaling-100km-to-3km-prmsl-output-loguniform-noise-pmin0.02"
CONFIG_FILENAME="train-100-to-3km-prmsl-output.yaml"

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
wandb_group=""

 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=4 # TODO: change to 8 after testing

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

IMAGE=$(cat $REPO_ROOT/latest_deps_only_image.txt)

gantry run \
    --name $JOB_NAME \
    --description 'Run downscaling 100km to 3km multivar training' \
    --workspace ai2/climate-titan \
    --priority urgent \
    --preemptible \
    --cluster ai2/jupiter \
    --cluster ai2/titan \
    --beaker-image $IMAGE \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP=$wandb_group \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-annak \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --weka climate-default:/climate-default \
    --gpus $N_GPUS \
    --shared-memory 400GiB \
    --budget ai2/climate \
    --no-conda \
    --install "pip install --no-deps ." \
    --allow-dirty \
    -- torchrun --nproc_per_node $N_GPUS -m fme.downscaling.train $CONFIG_PATH