#!/bin/bash
# uses the augusta cluster which doesn't have weka access but has GCS access and is
# typically more available than cirrascale clusters

set -e

JOB_NAME="eval-xshield-amip-100km-to-3km-conus-lower-sbound"
CONFIG_FILENAME="config-eval-conus.yaml"

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME

 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

N_NODES=4
NGPU=1

#IMAGE with B200 pytorch installed
IMAGE=01JWJ96JMF89D812JS159VF37N

#EXISTING_RESULTS_DATASET=01K8P3P5205396WR50FCMZR6P7 # best crps checkpoint from job using global validation
EXISTING_RESULTS_DATASET=01K8RWE83W8BEEAT2KRS94FVCD # best hist checkpoint from job using global validation

wandb_group=""

gantry run \
    --name $JOB_NAME \
    --description 'Run 100km to 3km evaluation on coarsened X-SHiELD' \
    --workspace ai2/ace \
    --priority high \
    --not-preemptible \
    --cluster ai2/ceres \
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
