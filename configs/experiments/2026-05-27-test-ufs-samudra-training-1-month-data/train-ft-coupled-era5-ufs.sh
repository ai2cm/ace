#!/bin/bash

set -e

JOB_NAME="test-ft-from-cm4-samudra-1pct-ocean-train-using-ufs-1day-dataset-20-step-mse"
JOB_GROUP="ufs-replay-ocean"
CONFIG_FILENAME="train-ft-from-cm4-to-ufs-test.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH="${SCRIPT_PATH}${CONFIG_FILENAME}"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=4
OCEAN_STATS_DATA="troya/2026-06-04-ufs-replay-ocean-1deg-19level-5day-1994-2023-stats"
ATMO_STATS_DATA="andrep/2026-03-19-era5-1deg-8layer-stats-1990-2019"
OCEAN_CKPT_DATASET="01KTA6CMTPNPYWD17JJ2E4XHMV"
ATMO_CKPT_DATASET="01KSVC6YS7C18SGYV4VPZYZ232"
cd $REPO_ROOT  # so config path is valid no matter where we are running this script

python -m fme.ace.validate_config --config_type train $CONFIG_PATH

gantry run \
    --name $JOB_NAME \
    --task-name $JOB_NAME \
    --description "SamudraI UFS pre-training" \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/climate-titan \
    --priority urgent \
    --preemptible \
    --cluster ai2/titan \
    --weka climate-default:/climate-default \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP=$JOB_GROUP \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset $STATS_DATA:/ocean_stats \
    --dataset $CKPT_DATASET:/weights \
    --gpus $N_GPUS \
    --shared-memory 400GiB \
    --budget ai2/atec-climate \
    --allow-dirty \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node $N_GPUS -m fme.coupled.train $CONFIG_PATH
