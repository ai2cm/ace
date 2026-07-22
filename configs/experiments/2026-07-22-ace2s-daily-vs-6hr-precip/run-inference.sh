#!/bin/bash

set -e

JOB_GROUP="ace2s-daily-vs-6hr-precip"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd "$REPO_ROOT"  # so config path is valid no matter where we are running this script

run_inference() {
  local config_filename="$1"
  local job_name="$2"
  local ckpt_dataset="$3"
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"

  python -m fme.ace.validate_config --config_type inference "$CONFIG_PATH"

  gantry run \
    --name "$job_name" \
    --task-name "$job_name" \
    --description 'ACE2S 10-day out-of-sample forecast starting 2020-07-15T06Z, PRATEsfc' \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace \
    --priority normal \
    --not-preemptible \
    --cluster ai2/titan \
    --cluster ai2/saturn \
    --cluster ai2/ceres \
    --env WANDB_USERNAME="$BEAKER_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP="$JOB_GROUP" \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "${ckpt_dataset}:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar" \
    --gpus 1 \
    --shared-memory 40GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- python -I -m fme.ace.inference "$CONFIG_PATH"
}

run_inference "inference-config-daily.yaml" "ace2s-daily-10day-precip-eval" "01KX95EEV3XD6KVTK25M0VJCKS"
run_inference "inference-config-6hr.yaml" "ace2s-6hr-10day-precip-eval" "01KXRJVGW1VFQP1XKSN0ZNGD57"
