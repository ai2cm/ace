#!/bin/bash

set -e

JOB_NAME_BASE="ace-aimip-fine-tune-decoder-pressure-levels"
JOB_GROUP="ace-aimip"
PRESSURE_LEVEL_SEPARATE_DECODER_LR_WARMUP_CONFIG_FILENAME="ace-fine-tune-pressure-level-separate-decoder-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
PRESSURE_LEVEL_SEPARATE_DECODER_LR_WARMUP_CONFIG_PATH=$SCRIPT_PATH/$PRESSURE_LEVEL_SEPARATE_DECODER_LR_WARMUP_CONFIG_FILENAME
EXISTING_RESULTS_DATASET="01K9B1MXD6V26S8BQH5CKY514C"  # best checkpoint is ace-aimip-train-rs3
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=4

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

python -m fme.ace.validate_config --config_type train $PRESSURE_LEVEL_SEPARATE_DECODER_LR_WARMUP_CONFIG_PATH

launch_job () {

    JOB_NAME=$1
    CONFIG_FILENAME=$2
    shift 2
    OVERRIDE="$@"

    gantry run \
        --name $JOB_NAME \
        --task-name $JOB_NAME \
        --description 'Fine-tune ACE decoder outputs on AIMIP period' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/climate-titan \
        --priority urgent \
        --preemptible \
        --cluster ai2/titan-cirrascale \
        --env WANDB_USERNAME=$BEAKER_USERNAME \
        --env WANDB_NAME=$JOB_NAME \
        --env WANDB_JOB_TYPE=training \
        --env WANDB_RUN_GROUP=$JOB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset brianhenn/era5-1deg-8layer-pressure-level-stats-1990-2019-v2:/statsdata \
        --dataset $EXISTING_RESULTS_DATASET:training_checkpoints/best_inference_ckpt.tar:/base_weights/ckpt.tar \
        --gpus $N_GPUS \
        --shared-memory 400GiB \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_FILENAME --override $OVERRIDE

}

# fine tune with separate decoder for pressure levels, with LR warmup
for SEED in 0 1 2 3; do
    JOB_NAME="${JOB_NAME_BASE}-separate-decoder-lr-warmup-RS${SEED}"
    OVERRIDE="seed=${SEED}"
    launch_job $JOB_NAME $PRESSURE_LEVEL_SEPARATE_DECODER_LR_WARMUP_CONFIG_PATH $OVERRIDE
done
