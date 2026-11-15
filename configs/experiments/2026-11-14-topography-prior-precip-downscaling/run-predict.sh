#!/bin/bash
# Run generation on 1 deg ERA5 with a trained checkpoint from this experiment.
# Usage: ./run-predict.sh <config filename> <job name> <training result dataset id>

set -e

CONFIG_FILENAME="$1"
JOB_NAME="$2"
CHECKPOINT_DATASET="$3"

if [ -z "$CONFIG_FILENAME" ] || [ -z "$JOB_NAME" ] || [ -z "$CHECKPOINT_DATASET" ]; then
    echo "Usage: $0 <config filename> <job name> <training result dataset id>"
    exit 1
fi

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME

WANDB_USERNAME=spencerc_ai2
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=8

cd $REPO_ROOT  # so config path is valid no matter where we are running this script
DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run \
    --name $JOB_NAME \
    --description 'Run 100km to 3km precipitation generation on 1deg ERA5' \
    --workspace ai2/ace \
    --priority low \
    --min-runtime 0 \
    --cluster ai2/jupiter \
    --beaker-image $DEPS_ONLY_IMAGE \
    --env WANDB_USERNAME=$WANDB_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP= \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset $CHECKPOINT_DATASET:checkpoints:/checkpoints \
    --weka climate-default:/climate-default \
    --gpus $N_GPUS \
    --shared-memory 400GiB \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node $N_GPUS -m fme.downscaling.predict $CONFIG_PATH
