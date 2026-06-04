#!/bin/bash

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
# since we use a service account API key for wandb, we use the beaker username to set the wandb username by default
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=1

cd "$REPO_ROOT"  # so config path is valid no matter where we are running this script

run_evaluator() {
  local config_filename="$1"
  local job_name="$2"
  local job_group="$3"
  local beaker_dataset_id="$4"
  local checkpoint_relpath="${5:-training_checkpoints/best_inference_ckpt.tar}"
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"

  python -m fme.ace.validate_config --config_type evaluator "$CONFIG_PATH"

  # Extract additional args from config header
  local extra_args=()
  while IFS= read -r line; do
    [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
  done < "$CONFIG_PATH"

  gantry run \
    --name "$job_name" \
    --task-name "$job_name" \
    --description 'Run ACE2S-ERA5 evaluator' \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace \
    --priority normal \
    --not-preemptible \
    --cluster ai2/saturn-cirrascale \
    --cluster ai2/ceres-cirrascale \
    --cluster ai2/titan \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP="$job_group" \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "${beaker_dataset_id}:${checkpoint_relpath}:/ckpt.tar" \
    --gpus "$N_GPUS" \
    --shared-memory 50GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    "${extra_args[@]}" \
    -- python -I -m fme.ace.evaluator "$CONFIG_PATH"
}

base_name="ace2s"
job_group="${base_name}-era5"

# Beaker result dataset ID from the training job (beaker experiment get ... | jq ...)


run_evaluator "ace-evaluator-config.yaml" \
  "$base_name-era5-eval-multi-step-ft-inverse-ace2-channel-weightings-rs0" \
  "$job_group" \
  "01KT9RWQH2C3JQ4EY4N3M5FYBJ"

run_evaluator "ace-evaluator-config.yaml" \
  "$base_name-era5-eval-multi-step-ft-no-var-weighting-rs0" \
  "$job_group" \
  "01KSVC6YS7C18SGYV4VPZYZ232"

