#!/bin/bash

# Run a 90-day abrupt-4xCO2 inference ensemble (36 initial
# conditions) with the ACE2-SOM paper checkpoint, saving daily-mean
# surface temperature to zarr.
# Checkpoint datasets are mounted via the "# arg:" header lines of each config.

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=1

cd "$REPO_ROOT"

run_inference() {
  local config_filename="$1"
  local job_name="$2"
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"

  python -m fme.ace.validate_config --config_type inference "$CONFIG_PATH"

  # Extract additional args from config header
  local extra_args=()
  while IFS= read -r line; do
    [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
  done < "$CONFIG_PATH"

  gantry run \
    --name "$job_name" \
    --description 'Run abrupt 4xCO2 ACE inference ensemble with daily surface temperature output' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority urgent \
    --min-runtime 2h \
    --cluster ai2/jupiter \
    --cluster ai2/ceres \
    --cluster ai2/saturn \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP= \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --gpus "$N_GPUS" \
    --shared-memory 20GiB \
    --weka climate-default:/climate-default \
    --system-python \
    --install "pip install --no-deps ." \
    "${extra_args[@]}" \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.ace.inference "$CONFIG_PATH"
}

run_inference "inference-ace2-som-paper-abrupt-4xCO2-daily-surface-temperature.yaml" "ace2-som-paper-abrupt-4xCO2-ensemble-daily-surface-temperature"
