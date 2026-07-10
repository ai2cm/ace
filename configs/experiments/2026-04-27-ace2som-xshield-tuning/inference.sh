#!/bin/bash

set -e

JOB_NAME="ace2s-era5-1step-tuned-xshield-inference-best-val-ckpt"
JOB_GROUP=""
#EXISTING_RESULTS_DATASET="01KWMYV98Q79G2FNY3CE95N2NG"  # tuned from SHiELD+
EXISTING_RESULTS_DATASET="01KWJRMVFPTCZFJEMAY9WVXNN7"  #  tuned from 1 step ERA5

CONFIG_FILENAME="inference-0k.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

#STATS_DATASET="annak/2026-04-27-vertically-resolved-1deg-c96-shield-ramped-climSST-random-CO2-ensemble-xshield-prmsl-stats"
STATS_DATASET="andrep/2026-06-08-vertically-resolved-1deg-c96-shield-ramped-climSST-random-CO2-ensemble-fme-dataset-stats"
cd $REPO_ROOT  # so config path is valid no matter where we are running this script
IMAGE="$(cat latest_deps_only_image.txt)"

#python -m fme.ace.validate_config --config_type evaluator $CONFIG_PATH

cd $REPO_ROOT && gantry run \
    --name $JOB_NAME \
    --task-name $JOB_NAME \
    --description 'Run ACE2S evaluator' \
    --beaker-image $IMAGE \
    --workspace ai2/downscaling \
    --priority high \
    --not-preemptible \
    --cluster ai2/titan \
    --cluster ai2/jupiter \
    --cluster ai2/ceres \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP=$JOB_GROUP \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-annak \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset $EXISTING_RESULTS_DATASET:training_checkpoints/best_ckpt.tar:/ckpt.tar \
    --dataset $STATS_DATASET:/statsdata \
    --gpus 1 \
    --shared-memory 50GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --no-python \
    --install "pip install --no-deps ." \
    --allow-dirty \
    -- python -I -m fme.ace.inference $CONFIG_PATH