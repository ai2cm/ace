#!/bin/bash

set -e

# JOB_NAME="evaluate-HiRO-xshield-amip-control-100km-to-3km-maritime-generate"

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

NGPU=8
IMAGE="$(cat $REPO_ROOT/latest_deps_only_image.txt)"

# HiROv1
EXISTING_RESULTS_DATASET=01K8RWE83W8BEEAT2KRS94FVCD # best hist checkpoint from job using global validation

# CONFIG_PATH="$SCRIPT_PATH/config-plus4k-tropic.yaml"
# JOB_NAME="evaluate-HiRO-xshield-amip-plus4K-100km-to-3km-maritime-generate"

CONFIG_PATH="$SCRIPT_PATH/config-control-tropic.yaml"
JOB_NAME="evaluate-HiRO-xshield-amip-control-100km-to-3km-maritime-generate-v2"

# HiROv1 fine-tuned
# EXISTING_RESULTS_DATASET=01KNM6H3JB1ZNS76HX17AAZRF7

# HiROv1 fine-tuned using log uniform noise distribution
# EXISTING_RESULTS_DATASET=01KNN2VFZC9AAK4NQR7QK5REDC

wandb_group=""

gantry run \
    --name $JOB_NAME \
    --description 'Run 100km to 3km evaluation on coarsened X-SHiELD' \
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
    -- torchrun --nproc_per_node $NGPU -m fme.downscaling.evaluator $CONFIG_PATH
