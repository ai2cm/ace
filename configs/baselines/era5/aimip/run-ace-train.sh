#!/bin/bash

set -e

JOB_NAME_BASE="ace-aimip-dataset-with-out-level0"
JOB_GROUP="ace-aimip-dataset-with-out-level0"
CONFIG_FILENAME="ace-train-no-strato-aimip-dataset.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=4

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

python -m fme.ace.validate_config --config_type train $CONFIG_PATH

launch_job () {

    JOB_NAME=$1
    shift
    OVERRIDE="$@"

    gantry run \
        --name $JOB_NAME \
        --task-name $JOB_NAME \
        --description 'Run ACE2-ERA5 training on AIMIP dataset with out level 0' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/climate-titan \
        --priority high \
        --preemptible \
        --cluster ai2/titan \
        --env WANDB_USERNAME=$BEAKER_USERNAME \
        --env WANDB_NAME=$JOB_NAME \
        --env WANDB_JOB_TYPE=training \
        --env WANDB_RUN_GROUP=$JOB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset andrep/era5-1deg-combined-aimip-forcing-1979-2014-stats:/statsdata \
        --gpus $N_GPUS \
        --shared-memory 400GiB \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --allow-dirty \
        --system-python \
        --install "pip install --no-deps ." \
        -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_PATH --override $OVERRIDE

}

# random seed ensemble
for SEED in 0; do
    JOB_NAME="${JOB_NAME_BASE}-rs${SEED}"
    OVERRIDE="seed=${SEED}"
    launch_job "$JOB_NAME" "$OVERRIDE"
done
