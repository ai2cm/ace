#!/bin/bash

set -e


SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')
CONFIG_PATH="$SCRIPT_PATH/finetune.yaml"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

NGPU=4
IMAGE="$(cat $REPO_ROOT/latest_deps_only_image.txt)"

# Previous model used for fine-tuning log-normal
# JOB_NAME="finetune-HiRO-v1-on-xshield-amip-plus4K-100km-to-3km"
# CHECKPOINT_DATASET="01KNJMW3F501NGTFRJTZE5MGP4"

JOB_NAME="finetune-HiRO-v1-log-uniform-on-xshield-amip-plus4K-100km-to-3km"
CHECKPOINT_DATASET="01KNJM638TD09QEE698NQCAFJC"


wandb_group=""

gantry run \
    --name $JOB_NAME \
    --description 'Run 100km to 3km fine-tuning on X-SHiELD +4K' \
    --workspace ai2/climate-titan \
    --priority urgent \
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
    --dataset $CHECKPOINT_DATASET:/previous_results/checkpoints \
    --weka climate-default:/climate-default \
    --gpus $NGPU \
    --shared-memory 400GiB \
    --budget ai2/climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node $NGPU -m fme.downscaling.train $CONFIG_PATH
