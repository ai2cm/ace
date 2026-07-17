#!/bin/bash

set -e

# Short segmented coupled inference run to reproduce
# https://github.com/ai2cm/ace/issues/471 : all segments end up named
# `<WANDB_NAME>-segment_0000` in wandb because os.environ["WANDB_NAME"] is only
# read by wandb.init on the first segment within a single process.
# 3 segments x 1 year each; expect distinct segment names but observe the bug.

JOB_NAME="cpl-segmented-inference-repro-471"
JOB_GROUP="2026-07-16-cpl-segement-inference"
EXISTING_RESULTS_DATASET="01KWW5C2CSBTZT5EQY7KBTC9FD"  # contains checkpoint for inference
CHECKPOINT_IN_DATASET="training_checkpoints/best_inference_ckpt.tar"
SEGMENTS=3
CONFIG_FILENAME="config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH="${SCRIPT_PATH}${CONFIG_FILENAME}"
 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

python -m fme.coupled.validate_config --config_type inference $CONFIG_PATH

gantry run \
    --name $JOB_NAME \
    --task-name $JOB_NAME \
    --description 'Segmented coupled inference' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority high \
    --cluster ai2/ceres \
    --cluster ai2/jupiter \
    --env WANDB_USERNAME=$WANDB_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP=$JOB_GROUP \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset $EXISTING_RESULTS_DATASET:$CHECKPOINT_IN_DATASET:/ckpt.tar \
    --gpus 1 \
    --shared-memory 50GiB \
    --weka climate-default:/climate-default \
    --budget ai2/climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- python -I -m fme.coupled.inference $CONFIG_PATH --segments $SEGMENTS
