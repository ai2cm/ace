#!/bin/bash

set -e

JOB_NAME_BASE="ace-foundation-model-era5-sst-pert"
JOB_GROUP="ace-foundation-model"
# this is from ace-aimip-fine-tune-decoder-pressure-levels-separate-decoder-lr-warmup-RS0
EXISTING_RESULTS_DATASET="01KKEXRZZDJJ0DJZ2KTSXYWAME"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
REPO_ROOT=$(git rev-parse --show-toplevel)

AIMIP_INFERENCE_P0K_CONFIG_FILENAME="ace-inference-era5-p0k.yaml"
AIMIP_INFERENCE_BASE_P0K_CONFIG_PATH=$SCRIPT_PATH/$AIMIP_INFERENCE_P0K_CONFIG_FILENAME
AIMIP_INFERENCE_P2K_CONFIG_FILENAME="ace-inference-era5-p2k.yaml"
AIMIP_INFERENCE_BASE_P2K_CONFIG_PATH=$SCRIPT_PATH/$AIMIP_INFERENCE_P2K_CONFIG_FILENAME
AIMIP_INFERENCE_P4K_CONFIG_FILENAME="ace-inference-era5-p4k.yaml"
AIMIP_INFERENCE_BASE_P4K_CONFIG_PATH=$SCRIPT_PATH/$AIMIP_INFERENCE_P4K_CONFIG_FILENAME

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

python -m fme.ace.validate_config --config_type inference $AIMIP_INFERENCE_BASE_P2K_CONFIG_PATH
python -m fme.ace.validate_config --config_type inference $AIMIP_INFERENCE_BASE_P4K_CONFIG_PATH

launch_job () {

    JOB_NAME=$1
    CONFIG_PATH=$2
    shift 2
    OVERRIDE="$@"

    cd $REPO_ROOT && gantry run \
        --name $JOB_NAME \
        --task-name $JOB_NAME \
        --description 'Run ACE2-ERA5 inference' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        --not-preemptible \
        --cluster ai2/ceres \
        --cluster ai2/titan \
        --cluster ai2/saturn \
        --cluster ai2/jupiter \
        --env WANDB_USERNAME=$BEAKER_USERNAME \
        --env WANDB_NAME=$JOB_NAME \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP=$JOB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $EXISTING_RESULTS_DATASET:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar \
        --gpus 1 \
        --shared-memory 50GiB \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- python -I -m fme.ace.inference $CONFIG_PATH

}

# same as above but use SST perturbed by +2K and +4K
JOB_NAME="${JOB_NAME_BASE}-p0k"
OVERRIDE=""
echo "Launching job: $JOB_NAME"
launch_job "$JOB_NAME" "$AIMIP_INFERENCE_BASE_P0K_CONFIG_PATH" "$OVERRIDE"

JOB_NAME="${JOB_NAME_BASE}-p2k"
OVERRIDE=""
echo "Launching job: $JOB_NAME"
launch_job "$JOB_NAME" "$AIMIP_INFERENCE_BASE_P2K_CONFIG_PATH" "$OVERRIDE"

JOB_NAME="${JOB_NAME_BASE}-p4k"
OVERRIDE=""
echo "Launching job: $JOB_NAME"
launch_job "$JOB_NAME" "$AIMIP_INFERENCE_BASE_P4K_CONFIG_PATH" "$OVERRIDE"
