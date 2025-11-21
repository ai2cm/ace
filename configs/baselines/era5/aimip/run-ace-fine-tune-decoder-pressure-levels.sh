#!/bin/bash

set -e

JOB_NAME_BASE="ace-aimip-fine-tune-decoder-pressure-levels"
JOB_GROUP="ace-aimip"
PRESSURE_LEVEL_CONFIG_FILENAME="ace-fine-tune-decoder-pressure-level-config.yaml"
PRESSURE_LEVEL_LR_WARMUP_CONFIG_FILENAME="ace-fine-tune-decoder-pressure-level-lr-warmup-config.yaml"
PRESSURE_LEVEL_REWEIGHT_CONFIG_FILENAME="ace-fine-tune-decoder-pressure-level-reweight-config.yaml"
PRESSURE_LEVEL_FROZEN_CONFIG_FILENAME="ace-fine-tune-decoder-pressure-level-frozen-config.yaml"
PRESSURE_LEVEL_FROZEN_LR_WARMUP_CONFIG_FILENAME="ace-fine-tune-decoder-pressure-level-frozen-lr-warmup-config.yaml"
PRESSURE_LEVEL_SEPARATE_DECODER_CONFIG_FILENAME="ace-fine-tune-decoder-pressure-level-separate-decoder-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
PRESSURE_LEVEL_CONFIG_PATH=$SCRIPT_PATH/$PRESSURE_LEVEL_CONFIG_FILENAME
PRESSURE_LEVEL_LR_WARMUP_CONFIG_PATH=$SCRIPT_PATH/$PRESSURE_LEVEL_LR_WARMUP_CONFIG_FILENAME
PRESSURE_LEVEL_REWEIGHT_CONFIG_PATH=$SCRIPT_PATH/$PRESSURE_LEVEL_REWEIGHT_CONFIG_FILENAME
PRESSURE_LEVEL_FROZEN_CONFIG_PATH=$SCRIPT_PATH/$PRESSURE_LEVEL_FROZEN_CONFIG_FILENAME
PRESSURE_LEVEL_FROZEN_LR_WARMUP_CONFIG_PATH=$SCRIPT_PATH/$PRESSURE_LEVEL_FROZEN_LR_WARMUP_CONFIG_FILENAME
PRESSURE_LEVEL_SEPARATE_DECODER_CONFIG_PATH=$SCRIPT_PATH/$PRESSURE_LEVEL_SEPARATE_DECODER_CONFIG_FILENAME
EXISTING_RESULTS_DATASET="01K9B1MXD6V26S8BQH5CKY514C"  # best checkpoint is ace-aimip-train-rs3
BEAKER_USERNAME=bhenn1983
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=4

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

python -m fme.ace.validate_config --config_type train $PRESSURE_LEVEL_CONFIG_PATH

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

# random seed ensemble of fine-tuning existing decoder to produce pressure level outputs
for SEED in 0 1 2 3; do
    JOB_NAME="${JOB_NAME_BASE}-RS${SEED}"
    OVERRIDE="seed=${SEED}"
    launch_job $JOB_NAME $PRESSURE_LEVEL_CONFIG_PATH $OVERRIDE
done

# same as above but with LR warmup
for SEED in 0 1; do
    JOB_NAME="${JOB_NAME_BASE}-lr-warmup-RS${SEED}"
    OVERRIDE="seed=${SEED}"
    launch_job $JOB_NAME $PRESSURE_LEVEL_LR_WARMUP_CONFIG_PATH $OVERRIDE
done

# same as above but smaller ensemble with downweighted q1/q2/q3/q4 to avoid overfitting
for SEED in 0 1; do
    JOB_NAME="${JOB_NAME_BASE}-downweight-q-RS${SEED}"
    OVERRIDE="seed=${SEED} \
stepper.loss.weights.specific_total_water_1=0.1 \
stepper.loss.weights.specific_total_water_2=0.25 \
stepper.loss.weights.specific_total_water_3=0.5 \
stepper.loss.weights.specific_total_water_4=0.5"
    launch_job $JOB_NAME $PRESSURE_LEVEL_CONFIG_PATH $OVERRIDE
done

# fine tune with unfrozen decoder, but reweight (heavily downweight the fine-tuned variables)
for SEED in 0 1; do
    JOB_NAME="${JOB_NAME_BASE}-reweight-RS${SEED}"
    OVERRIDE="seed=${SEED}"
    launch_job $JOB_NAME $PRESSURE_LEVEL_REWEIGHT_CONFIG_PATH $OVERRIDE
done

# random seed ensemble of fine-tuning existing decoder to produce pressure level outputs, new weights only
for SEED in 0 1; do
    JOB_NAME="${JOB_NAME_BASE}-frozen-RS${SEED}"
    OVERRIDE="seed=${SEED}"
    launch_job $JOB_NAME $PRESSURE_LEVEL_FROZEN_CONFIG_PATH $OVERRIDE
done

# same as above but with LR warmup
for SEED in 0 1; do
    JOB_NAME="${JOB_NAME_BASE}-frozen-lr-warmup-RS${SEED}"
    OVERRIDE="seed=${SEED}"
    launch_job $JOB_NAME $PRESSURE_LEVEL_FROZEN_LR_WARMUP_CONFIG_PATH $OVERRIDE
done

# fine tune with separate decoder for pressure levels
for SEED in 0 1; do
    JOB_NAME="${JOB_NAME_BASE}-separate-decoder-RS${SEED}"
    OVERRIDE="seed=${SEED}"
    launch_job $JOB_NAME $PRESSURE_LEVEL_SEPARATE_DECODER_CONFIG_PATH $OVERRIDE
done
