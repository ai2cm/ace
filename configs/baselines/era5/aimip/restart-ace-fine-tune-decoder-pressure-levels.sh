#!/bin/bash

set -e

JOB_NAME_BASE="ace-aimip-fine-tune-decoder-pressure-levels"
JOB_GROUP="ace-aimip"
PRESSURE_LEVEL_CONFIG_FILENAME="ace-fine-tune-decoder-pressure-level-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
PRESSURE_LEVEL_CONFIG_PATH=$SCRIPT_PATH/$PRESSURE_LEVEL_CONFIG_FILENAME
PRETRAINED_CHECKPOINT_DATASET="01K9B1MXD6V26S8BQH5CKY514C"  # best checkpoint is ace-aimip-train-rs3
BEAKER_USERNAME=bhenn1983
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=4

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

python -m fme.ace.validate_config --config_type train $PRESSURE_LEVEL_CONFIG_PATH

launch_job () {

    JOB_NAME=$1
    PREVIOUS_RESULTS_DATASET=$2
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
        --dataset $PRETRAINED_CHECKPOINT_DATASET:training_checkpoints/best_inference_ckpt.tar:/base_weights/ckpt.tar \
        --dataset $PREVIOUS_RESULTS_DATASET:/existing_results_dir \
        --gpus $N_GPUS \
        --shared-memory 400GiB \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train $PRESSURE_LEVEL_CONFIG_PATH --override $OVERRIDE

}

# add 120 more epochs
JOB_NAME="${JOB_NAME_BASE}-RS0"
PREVIOUS_RESULTS_DATASET="01KA2F5J9768HR54369MKEHYB4"
OVERRIDE="seed=0 resume_results={existing_dir: /existing_results_dir, resume_wandb: true}"
launch_job "$JOB_NAME" "$PREVIOUS_RESULTS_DATASET" "$OVERRIDE"

JOB_NAME="${JOB_NAME_BASE}-RS1"
PREVIOUS_RESULTS_DATASET="01KADMD5RAEANTP3M6GWZTXNXA"
OVERRIDE="seed=1 resume_results={existing_dir: /existing_results_dir, resume_wandb: true}"
launch_job "$JOB_NAME" "$PREVIOUS_RESULTS_DATASET" "$OVERRIDE"

JOB_NAME="${JOB_NAME_BASE}-RS2"
PREVIOUS_RESULTS_DATASET="01KAC0B5ZC96GVJV46HNK9QZ9X"
OVERRIDE="seed=2 resume_results={existing_dir: /existing_results_dir, resume_wandb: true}"
launch_job "$JOB_NAME" "$PREVIOUS_RESULTS_DATASET" "$OVERRIDE"

JOB_NAME="${JOB_NAME_BASE}-RS3"
PREVIOUS_RESULTS_DATASET="01KA2F5Q45122PYGW4NZBME01V"
OVERRIDE="seed=3 resume_results={existing_dir: /existing_results_dir, resume_wandb: true}"
launch_job "$JOB_NAME" "$PREVIOUS_RESULTS_DATASET" "$OVERRIDE"
