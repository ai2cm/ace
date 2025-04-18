#!/bin/bash
# uses the augusta cluster which doesn't have weka access but has GCS access and is
# typically more available than cirrascale clusters

set -e

# for now, just using a single node
NGPU=8


JOB_NAME="eval-conus-downscaling-25km-to-3km-emamodel-v1"  # recommended but not required to change this
CONFIG_FILENAME="eval-conus-composite-25km-to-3km-augusta.yaml"

EXISTING_RESULTS_DATASET="01JS04B574CGDSB2DHP394XNVX"  # 25km-to-3km https://beaker.allen.ai/orgs/ai2/workspaces/downscaling/work/01JRWXTXCM8NJQ334MK6A3HFVY

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME

 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script
DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run \
    --name $JOB_NAME \
    --description 'Run downscaling 25km to 3km evaluation' \
    --workspace ai2/downscaling \
    --priority normal \
    --preemptible \
    --cluster ai2/augusta-google-1 \
    --gpus $NGPU \
    --budget ai2/climate \
    --beaker-image $DEPS_ONLY_IMAGE \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP= \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset $EXISTING_RESULTS_DATASET:checkpoints:/checkpoints \
    --no-conda \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node $NGPU -m fme.downscaling.evaluator $CONFIG_PATH