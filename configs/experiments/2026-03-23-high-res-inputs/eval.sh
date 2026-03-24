#!/bin/bash

set -e

#JOB_NAME="eval-xshield-amip-100km-to-3km-0.5sigmaexp-tropics-events"
JOB_NAME="eval-debug-event-channel-mixing"

CONFIG_FILENAME="debug-events.yaml"

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME

 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

N_NODES=1
NGPU=2

IMAGE="$(cat latest_deps_only_image.txt)"

EXISTING_RESULTS_DATASET_WINDSPRMSL=01KKC6TPBFKV4MC5HDNFD3P0HB
EXISTING_RESULTS_DATASET_PRATE=01KMEEQ0ER0MRVSXA3CK4E47V7
wandb_group=""

#--not-preemptible \
#     --dataset $EXISTING_RESULTS_DATASET:checkpoints:/checkpoints \

#    --dataset $EXISTING_RESULTS_DATASET:hiro-public-ckpt.tar:/checkpoints/best.ckpt \
#    --dataset $EXISTING_RESULTS_DATASET_PRATE:checkpoints:/prate  \


gantry run \
    --name $JOB_NAME \
    --description 'Run 100km to 3km evaluation on coarsened X-SHiELD' \
    --workspace ai2/climate-titan \
    --priority urgent \
    --cluster ai2/jupiter \
    --cluster ai2/titan \
    --beaker-image $IMAGE \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP=$wandb_group \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-annak \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset $EXISTING_RESULTS_DATASET_WINDSPRMSL:checkpoints:/winds_prmsl  \
    --weka climate-default:/climate-default \
    --gpus $NGPU \
    --shared-memory 400GiB \
    --budget ai2/climate \
    --no-conda \
    --install "pip install --no-deps ." \
    --allow-dirty \
    -- torchrun --nproc_per_node $NGPU -m fme.downscaling.evaluator $CONFIG_PATH
