#!/bin/bash

set -e

JOB_NAME_BASE="ace-aimip-evaluator-seed-selection"
JOB_GROUP="ace-aimip"
SEED_CHECKPOINT_IDS=("01K9B1MR70QWN90KNY7NM22K5M" \
  "01K9B1MT4QY1ZEZPPS53G2SXPK" \
  "01K9B1MVP3VS3NEABHT0W151AX" \
  "01K9B1MXD6V26S8BQH5CKY514C" \
  )
FINE_TUNED_SEPARATE_DECODER_CHECKPOINT_IDS=("01KAKXY0EK24K7BZK2N8SPJ5SJ"\
  "01KAVVAKANNYY096MYCGSZ7RMQ" \
  "01KAVVGKY28P5N1VA883C63EBY" \
  "01KAVVN8YPPB3P6ZSD0BGCCVX7"
)
CONFIG_FILENAME="ace-evaluator-seed-selection-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

python -m fme.ace.validate_config --config_type evaluator $CONFIG_PATH

launch_job () {
    local JOB_NAME=$1
    local SEED_CHECKPOINT_ID=$2

    cd $REPO_ROOT && gantry run \
        --name $JOB_NAME \
        --task-name $JOB_NAME \
        --description 'Run ACE2-ERA5 evaluator for AIMIP seed selection' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        --not-preemptible \
        --cluster ai2/saturn-cirrascale \
        --cluster ai2/ceres-cirrascale \
        --cluster ai2/titan-cirrascale \
        --cluster ai2/jupiter-cirrascale-2 \
        --env WANDB_USERNAME=$BEAKER_USERNAME \
        --env WANDB_NAME=$JOB_NAME \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP=$JOB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $SEED_CHECKPOINT_ID:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar \
        --gpus 1 \
        --shared-memory 100GiB \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- python -I -m fme.ace.evaluator $CONFIG_PATH

    }

# pre-trained
for (( i=0; i<${#SEED_CHECKPOINT_IDS[@]}; i++ )); do
    JOB_NAME="$JOB_NAME_BASE-RS$i"
    echo "Launching job for seed $i checkpoint ID: ${SEED_CHECKPOINT_IDS[$i]}"
    launch_job "$JOB_NAME" "${SEED_CHECKPOINT_IDS[$i]}"
done

# fine-tuned with separate decoder
for  (( i=0; i<${#FINE_TUNED_SEPARATE_DECODER_CHECKPOINT_IDS[@]}; i++ )); do
    JOB_NAME="$JOB_NAME_BASE-RS3-pressure-level-fine-tuned-separate-decoder-RS$i"
    echo "Launching job for fine-tuned with separate decoder seed $i checkpoint ID: ${FINE_TUNED_SEPARATE_DECODER_CHECKPOINT_IDS[$i]}"
    launch_job "$JOB_NAME" "${FINE_TUNED_SEPARATE_DECODER_CHECKPOINT_IDS[$i]}"
done
