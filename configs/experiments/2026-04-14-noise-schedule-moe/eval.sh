#!/bin/bash

set -e

JOB_NAME="eval-tmp2m-prate-events"
#JOB_NAME="eval-global-trained-denoising-moe-events"

CONFIG_FILENAME="eval-tmp2m-prate-events.yaml"

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME

 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

N_NODES=1
NGPU=2

IMAGE="$(cat latest_deps_only_image.txt)"

EXISTING_RESULTS_DATASET_HIGH_SIGMA=01KRPP269PFM7G9EY8BWZ3AW04
EXISTING_RESULTS_DATASET_LOW_SIGMA=01KTCYVZBJJJ7C8K81C0VG0ZNB
EXISTING_RESULTS_DATASET=01KTMJ8V4RRRVYGBJKEJA9Y0VM
wandb_group=""

#--not-preemptible \
#     --dataset $EXISTING_RESULTS_DATASET:checkpoints:/checkpoints \

#    --dataset $EXISTING_RESULTS_DATASET:hiro-public-ckpt.tar:/checkpoints/best.ckpt \
#    --dataset $EXISTING_RESULTS_DATASET_HIGH_SIGMA:checkpoints:/checkpoints_high_sigma  \
#    --dataset $EXISTING_RESULTS_DATASET_LOW_SIGMA:checkpoints:/checkpoints_low_sigma  \

gantry run \
    --name $JOB_NAME \
    --description 'Run 100km to 3km evaluation on coarsened X-SHiELD' \
    --workspace ai2/climate-titan \
    --priority urgent \
    --cluster ai2/titan \
    --beaker-image $IMAGE \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP=$wandb_group \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-annak \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset $EXISTING_RESULTS_DATASET:checkpoints:/checkpoints \
    --weka climate-default:/climate-default \
    --gpus $NGPU \
    --shared-memory 400GiB \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    --allow-dirty \
    -- torchrun --nproc_per_node $NGPU -m fme.downscaling.evaluator $CONFIG_PATH
