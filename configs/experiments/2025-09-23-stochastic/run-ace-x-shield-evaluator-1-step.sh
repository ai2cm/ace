#!/bin/bash

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=1

cd "$REPO_ROOT"

run_eval() {
  local config_filename="$1"
  local job_name="$2"
  local start_step="${3:-0}"
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"

  python -m fme.ace.validate_config --config_type evaluator "$CONFIG_PATH"

  # Extract additional args from config header
  local extra_args=()
  while IFS= read -r line; do
    [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
  done < "$CONFIG_PATH"

  echo "Starting evaluator job: $job_name for start step: $start_step"
  gantry run \
    --name "$job_name" \
    --description 'Run ACE2-ERA5 evaluator' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority normal \
    --not-preemptible \
    --cluster ai2/titan \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP= \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --gpus 1 \
    --shared-memory 200GiB \
    --weka climate-default:/climate-default \
    --budget ai2/climate \
    --system-python \
    --allow-dirty \
    --install "pip install --no-deps ." \
    "${extra_args[@]}" \
    -- python -I -m fme.ace.evaluator "$CONFIG_PATH" \
    --override loader.start_indices.first=$start_step
}

base_name="stochastic"
for i in {0..1459..146}; do
  start_step=$i
  start_step=$((start_step + 14608)) #offset to match val / test period
  end_step=$((start_step + 146))
  run_eval "ace-x-shield-eval-config-rs0-1-year-1-steps-only.yaml" "$base_name-x-shield-n384-e1c9-era5-ft-20step-v2-eval-1-step-${start_step}-${end_step}" "$start_step"
done
