#!/bin/bash
# uses the augusta cluster which doesn't have weka access but has GCS access and is
# typically more available than cirrascale clusters

set -e

JOB_NAME="eval-xshield-amip-100km-to-3km-cascade-global"
CONFIG_FILENAME="config-generate-on-perfect-pred-global.yaml"

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME

 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

N_NODES=1
NGPU=2

#IMAGE with B200 pytorch installed
IMAGE=01JWJ96JMF89D812JS159VF37N
EXISTING_RESULTS_DATASET_100_25="01K6B4RB7810TBDBFSMWQBBC1E"  # 100km-to-25kmhttps://beaker.allen.ai/orgs/ai2/workspaces/downscaling/work/01JXZZ1J7MATW62YHKYEYGEB3Z
EXISTING_RESULTS_DATASET_25_3="01K6YHQT5B72H8NYA0E93AN4R6"  # 25km-to-3km https://beaker.allen.ai/orgs/ai2/workspaces/downscaling/work/01JYFNSDT54K16D11VKZTHK45V
wandb_group=""

gantry run \
    --name $JOB_NAME \
    --description 'Run 100km to 3km evaluation on coarsened X-SHiELD' \
    --workspace ai2/downscaling \
    --priority low \
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
    --dataset $EXISTING_RESULTS_DATASET_100_25:checkpoints:/checkpoints_25 \
    --dataset $EXISTING_RESULTS_DATASET_25_3:checkpoints:/checkpoints_3 \
    --weka climate-default:/climate-default \
    --gpus $NGPU \
    --shared-memory 400GiB \
    --budget ai2/climate \
    --no-conda \
    --install "pip install --no-deps ." \
    --allow-dirty \
    -- torchrun --nproc_per_node $NGPU -m fme.downscaling.evaluator $CONFIG_PATH
