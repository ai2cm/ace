#!/bin/bash
# Submit downscaling jobs for split tc_tracks_predict configs.

set -e

RANGE_START=15
RANGE_END=15

JOB_NAME_BASE="predict-ace2s-xshield-tc-tracks"
#JOB_NAME="eval-global-trained-denoising-moe-events"

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')
CONFIG_DIR=$SCRIPT_PATH

 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

N_NODES=1
NGPU=1

IMAGE="$(cat latest_deps_only_image.txt)"

#EXISTING_RESULTS_DATASET=01KNM6H3JB1ZNS76HX17AAZRF7

EXISTING_RESULTS_DATASET_HIGH_SIGMA=01KRGZT4X2QCW2RFH7WN7X8BYA
EXISTING_RESULTS_DATASET_LOW_SIGMA=01KRBYGNYJ6FD7PGNF3VVHQ5V1


wandb_group=""

#     --dataset $EXISTING_RESULTS_DATASET:checkpoints:/checkpoints \

#    --dataset $EXISTING_RESULTS_DATASET:hiro-public-ckpt.tar:/checkpoints/best.ckpt \
#--not-preemptible \

for i in $(seq "$RANGE_START" "$RANGE_END"); do
    #CONFIG_PATH="$CONFIG_DIR/tc_tracks_predict_${i}.yaml"
    CONFIG_PATH="$CONFIG_DIR/test_tc_track.yaml"
    if [[ ! -f "$CONFIG_PATH" ]]; then
        echo "Config not found: $CONFIG_PATH" >&2
        exit 1
    fi
    #JOB_NAME="${JOB_NAME_BASE}-${i}"
    JOB_NAME="predict-ace2s-xshield-tc-tracks-test"
    gantry run \
        --name $JOB_NAME \
        --description 'Run 100km to 3km evaluation on ACE2S-SHiELD' \
        --workspace ai2/climate-titan \
        --priority urgent \
        --cluster ai2/titan \
        --beaker-image $IMAGE \
        --env WANDB_USERNAME=$BEAKER_USERNAME \
        --env WANDB_NAME=$JOB_NAME \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP=$wandb_group \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-annak \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $EXISTING_RESULTS_DATASET_HIGH_SIGMA:checkpoints:/checkpoints_high_sigma  \
        --dataset $EXISTING_RESULTS_DATASET_LOW_SIGMA:checkpoints:/checkpoints_low_sigma  \
        --weka climate-default:/climate-default \
        --gpus $NGPU \
        --shared-memory 400GiB \
        --budget ai2/atec-climate \
        --no-python \
        --install "pip install --no-deps ." \
        --allow-dirty \
        -- torchrun --nproc_per_node $NGPU -m fme.downscaling.predict $CONFIG_PATH
done
