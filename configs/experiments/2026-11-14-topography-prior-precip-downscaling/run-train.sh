#!/bin/bash
# Train a topography-conditioned 3 km precipitation prior. Data is read from
# weka (/climate-default), so this must run on a cluster with weka access.

set -e

JOB_NAME="topography-prior-precip-100km-to-3km-train"
CONFIG_FILENAME="train.yaml"

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME

# since we use a service account API key for wandb, we use the beaker username to set the wandb username
WANDB_USERNAME=spencerc_ai2
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=8

cd $REPO_ROOT  # so config path is valid no matter where we are running this script
DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run \
    --name $JOB_NAME \
    --description 'Train topography-conditioned 3km precipitation prior for block-mean projected downscaling' \
    --workspace ai2/ace \
    --priority low \
    --min-runtime 0 \
    --preemptible \
    --cluster ai2/jupiter \
    --cluster ai2/ceres \
    --beaker-image $DEPS_ONLY_IMAGE \
    --env WANDB_USERNAME=$WANDB_USERNAME
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP= \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --weka climate-default:/climate-default \
    --gpus $N_GPUS \
    --shared-memory 400GiB \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node $N_GPUS -m fme.downscaling.train $CONFIG_PATH
