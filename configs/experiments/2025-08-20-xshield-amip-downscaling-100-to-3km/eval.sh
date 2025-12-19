#!/bin/bash
# uses the augusta cluster which doesn't have weka access but has GCS access and is
# typically more available than cirrascale clusters

set -e

JOB_NAME="generate-xshield-amip-events-with-static-inputs-wind-only-churn1"
CONFIG_FILENAME="eval-wind-events.yaml"

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME

 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

N_NODES=1
NGPU=2

IMAGE=spencerc/fme-deps-only-0196723e

#EXISTING_RESULTS_DATASET=01K9PGGJSSKHQV4EKN4HMCERXB  # best hist checkpoint from multivar without static inputs
#EXISTING_RESULTS_DATASET=01KBG9EYTQNR082TWYFW5XWW3J  # best hist checkpoint from multivar with static inputs
EXISTING_RESULTS_DATASET=01KCG9JWS8QEAAHMCRJY6EGW68  # best hist checkpoint from wind-only model with static inputs
wandb_group=""

gantry run \
    --name $JOB_NAME \
    --description 'Run 100km to 3km evaluation on coarsened X-SHiELD' \
    --workspace ai2/ace \
    --priority high \
    --not-preemptible \
    --cluster ai2/titan \
    --cluster ai2/jupiter \
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
    --no-conda \
    --install "pip install --no-deps ." \
    --allow-dirty \
    -- torchrun --nproc_per_node $NGPU -m fme.downscaling.evaluator $CONFIG_PATH