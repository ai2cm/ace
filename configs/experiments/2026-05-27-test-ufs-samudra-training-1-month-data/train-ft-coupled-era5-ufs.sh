#!/bin/bash

set -e

JOB_NAME="ft-coupled-era5-ufs-stochastic-5day-ocean"
JOB_GROUP="ufs-replay-ocean"
CONFIG_FILENAME="train-ft-coupled-era5-ufs.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH="${SCRIPT_PATH}${CONFIG_FILENAME}"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=4
OCEAN_STATS_DATA="troya/2026-06-04-ufs-replay-ocean-1deg-19level-5day-1994-2023-stats"
ATMO_STATS_DATA="andrep/2026-03-19-era5-1deg-8layer-stats-1990-2019"
OCEAN_CKPT_DATASET="01KTA6CMTPNPYWD17JJ2E4XHMV"
ATMO_CKPT_DATASET="01KSVC6YS7C18SGYV4VPZYZ232"
OCEAN_CKPT="best_inference_ckpt"
ATMO_CKPT="best_inference_ckpt"
cd $REPO_ROOT  # so config path is valid no matter where we are running this script

python -m fme.coupled.validate_config "$CONFIG_PATH" --config_type train

gantry run \
    --name $JOB_NAME \
    --task-name $JOB_NAME \
    --description "Coupled ERA5+UFS ensemble fine-tuning" \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority high \
    --preemptible \
    --cluster ai2/titan \
    --weka climate-default:/climate-default \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP=$JOB_GROUP \
    --env FME_LOG_NAN_DIAGNOSTICS=1 \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset $OCEAN_STATS_DATA:/ocean_stats \
    --dataset $ATMO_STATS_DATA:/atmos_stats \
    --dataset "$OCEAN_CKPT_DATASET:training_checkpoints/${OCEAN_CKPT}.tar:/ocean_ckpt.tar" \
    --dataset "$ATMO_CKPT_DATASET:training_checkpoints/${ATMO_CKPT}.tar:/atmos_ckpt.tar" \
    --gpus $N_GPUS \
    --shared-memory 400GiB \
    --budget ai2/atec-climate \
    --allow-dirty \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node $N_GPUS -m fme.coupled.train "$CONFIG_PATH"
