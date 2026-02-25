#!/bin/bash

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=4

cd "$REPO_ROOT"

run_training() {
  local config_filename="$1"
  local job_name="$2"
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"

  python -m fme.ace.validate_config --config_type train "$CONFIG_PATH"

  # Extract additional args from config header
  local extra_args=()
  while IFS= read -r line; do
    [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
  done < "$CONFIG_PATH"

  gantry run \
    --name "$job_name" \
    --description 'Run ACE training' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/climate-titan \
    --priority urgent \
    --preemptible \
    --cluster ai2/titan \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP= \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --gpus "$N_GPUS" \
    --shared-memory 400GiB \
    --weka climate-default:/climate-default \
    --budget ai2/climate \
    --system-python \
    --allow-dirty \
    --install "pip install --no-deps ." \
    "${extra_args[@]}" \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.ace.train "$CONFIG_PATH"
}

run_training "train-amip-c96-shield-ace2.yaml" "train-amip-c96-shield-ace2"
# run_training "train-amip-c96-shield-daily.yaml" "train-amip-c96-shield-daily"
# run_training "train-amip-c96-shield-daily-pos16.yaml" "train-amip-c96-shield-daily-pos16"
# run_training "train-amip-c96-shield-daily-pos16-rsop.yaml" "train-amip-c96-shield-daily-pos16-rsop"
# run_training "train-amip-c96-shield-daily-e1c7d2-pos16-rsop.yaml" "train-amip-c96-shield-daily-e1c7d2-pos16-rsop"
# run_training "train-amip-c96-shield-daily-e1c7d2-l2-pos16-rsop.yaml" "train-amip-c96-shield-daily-e1c7d2-l2-pos16-rsop"
# run_training "train-amip-c96-shield-daily-e1c7l2-pos16-rsop.yaml" "train-amip-c96-shield-daily-e1c7l2-pos16-rsop"
# run_training "train-amip-c96-shield.yaml" "train-amip-c96-shield"
# run_training "train-amip-c96-shield-rsop.yaml" "train-amip-c96-shield-rsop"
