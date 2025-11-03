#!/bin/bash

set -e

JOB_NAME="generate-xshield-amip-100km-to-3km-zarr-outputs-cascade-test"
CONFIG_FILENAME="config-generate-cascade.yaml"

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME

 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

NGPU=2
IMAGE="$(cat $REPO_ROOT/latest_deps_only_image.txt)"

EXISTING_RESULTS_DATASET_100_25="01K6B4RB7810TBDBFSMWQBBC1E"  # 100km-to-25km https://beaker.allen.ai/orgs/ai2/workspaces/climate-ceres/work/01K61ENSWYD3VCW90AX32QYJQ4?taskId=01K61ENSX4MPZVDJ9S7YC1Z2YD&jobId=01K6B4RBC2B2ZNQ09JPCRQ23DN
EXISTING_RESULTS_DATASET_25_3="01K6YHQT5B72H8NYA0E93AN4R6"  # 25km-to-3km https://beaker.allen.ai/orgs/ai2/workspaces/climate-ceres/work/01K6YHQT4ZBF3CG56P388E09HD?taskId=01K6YHQT55JAF2875YPSPE4XM3&jobId=01K6YHQT8TVEA1CHCC26GZWNX2


wandb_group=""

gantry run \
    --name $JOB_NAME \
    --description 'Run 100km to 3km generation on coarsened X-SHiELD' \
    --workspace ai2/downscaling \
    --priority high \
    --not-preemptible \
    --cluster ai2/titan \
    --beaker-image $IMAGE \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP=$wandb_group \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset $EXISTING_RESULTS_DATASET_100_25:checkpoints:/checkpoints_25 \
    --dataset $EXISTING_RESULTS_DATASET_25_3:checkpoints:/checkpoints_3 \
    --weka climate-default:/climate-default \
    --gpus $NGPU \
    --shared-memory 400GiB \
    --budget ai2/climate \
    --no-conda \
    --install "pip install --no-deps ." \
    --allow-dirty \
    -- torchrun --nproc_per_node $NGPU -m fme.downscaling.generate $CONFIG_PATH
