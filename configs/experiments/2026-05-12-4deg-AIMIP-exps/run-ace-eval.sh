#!/bin/bash

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
 # since we use a service account API key for wandb, we use the beaker username to set the wandb username by default
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

run_evaluation() {
  local config_filename="$1"
  local job_name="$2"
  local job_group="$3"
  local existing_results_dataset="$4"
  local CONFIG_PATH="$SCRIPT_PATH$config_filename"

  python -m fme.ace.validate_config --config_type evaluator "$CONFIG_PATH"

  gantry run \
    --name "$job_name" \
    --task-name "$job_name" \
    --description 'Run ACE2-ERA5 evaluator' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority normal \
    --not-preemptible \
    --cluster ai2/saturn-cirrascale \
    --cluster ai2/ceres-cirrascale \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP="$job_group" \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "$existing_results_dataset:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar" \
    --gpus 1 \
    --shared-memory 50GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- python -I -m fme.ace.evaluator "$CONFIG_PATH"
}

run_evaluation "${1:-ace-eval-config-4deg-AIMIP.yaml}" "${2:-ace-eval-config-4deg-AIMIP}" "${3:-ace2-era5}" "${4:-01KS18SSSW3G10N75YR9HF78RW}"
