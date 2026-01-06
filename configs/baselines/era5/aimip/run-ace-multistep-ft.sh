#!/bin/bash

set -e

JOB_NAME_BASE="ace2.1s-aimip-multistep-ft"
JOB_GROUP="ace-aimip"
CONFIG_FILENAME="ace-multistep-ft-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
BEAKER_USERNAME=bhenn1983
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=4

PRETRAINED_MODELS=("01KE3ZN6G8NQK25GHFYJSV55WW" "01KE3ZN97Y2NANSV0EGMBC6218")

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

python -m fme.ace.validate_config --config_type train $CONFIG_PATH

launch_job () {

    JOB_NAME=$1
    PRETRAINED_MODEL=$2
    shift 2
    OVERRIDE="$@"

    gantry run \
        --name $JOB_NAME \
        --task-name $JOB_NAME \
        --description 'Run ACE2-ERA5 multistep fine-tuning on AIMIP period' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        --preemptible \
        --cluster ai2/titan-cirrascale \
        --env WANDB_USERNAME=$BEAKER_USERNAME \
        --env WANDB_NAME=$JOB_NAME \
        --env WANDB_JOB_TYPE=training \
        --env WANDB_RUN_GROUP=$JOB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset oliverwm/era5-1deg-8layer-stats-1990-2019-v2:/statsdata \
        --dataset $PRETRAINED_MODEL:/weights \
        --gpus $N_GPUS \
        --shared-memory 400GiB \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_PATH --override $OVERRIDE

}

# pretrained model checkpoints
for ((i = 0; i < ${#PRETRAINED_MODELS[@]}; i++)); do
    JOB_NAME="${JOB_NAME_BASE}-rs${i}"
    PRETRAINED_MODEL="${PRETRAINED_MODELS[$i]}"
    launch_job "$JOB_NAME" "$PRETRAINED_MODEL"
done
