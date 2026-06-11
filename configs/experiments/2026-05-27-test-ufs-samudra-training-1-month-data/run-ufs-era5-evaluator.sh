#!/bin/bash

set -e

JOB_GROUP="ufs-replay-ocean"
BASE_JOB_NAME="ft-coupled-era5-ufs-stochastic-5day-ocean-evaluator"
EXISTING_RESULTS_DATASET="01KTMD1TRMSFN4FHP19CZAST62"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd "$REPO_ROOT"  # so config path is valid no matter where we are running this script

run_eval() {
  local config_filename="$1"
  local job_name="$2"
  local ckpt_dataset="$3"
  local CONFIG_PATH="${SCRIPT_PATH}${config_filename}"

  python -m fme.coupled.validate_config --config_type evaluator "$CONFIG_PATH"

  gantry run \
    --name "$job_name" \
    --task-name "$job_name" \
    --description "Coupled ERA5+UFS ensemble fine-tuning evaluator" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace \
    --priority high \
    --not-preemptible \
    --cluster ai2/titan \
    --cluster ai2/saturn \
    --cluster ai2/ceres \
    --weka climate-default:/climate-default \
    --env WANDB_USERNAME="$BEAKER_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP="$JOB_GROUP" \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "${ckpt_dataset}:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar" \
    --gpus 1 \
    --shared-memory 50GiB \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- python -I -m fme.coupled.evaluator "$CONFIG_PATH"
}

for year in $(seq 2010 2020); do
    year_str=$(printf "%04d" "$year")

    run_eval "./configs/ufs-era5-fully-coupled-v0/evaluator-config-coupled-ERA5-UFS-v0-yr${year_str}.yaml" \
             "${BASE_JOB_NAME}-yr${year_str}" \
             "$EXISTING_RESULTS_DATASET"
done
