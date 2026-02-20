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
  local ckpt_dataset="$3"
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"

  python -m fme.coupled.validate_config --config_type evaluator "$CONFIG_PATH"

  # Extract additional args from config header
  local extra_args=()
  while IFS= read -r line; do
    [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
  done < "$CONFIG_PATH"

  gantry run \
    --name "$job_name" \
    --description 'Run SamudrACE evaluator' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority high \
    --not-preemptible \
    --cluster ai2/titan \
    --cluster ai2/saturn \
    --cluster ai2/ceres \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP= \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "${ckpt_dataset}:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar" \
    --gpus 1 \
    --shared-memory 200GiB \
    --weka climate-default:/climate-default \
    --budget ai2/climate \
    --system-python \
    --allow-dirty \
    --install "pip install --no-deps ." \
    "${extra_args[@]}" \
    -- python -I -m fme.coupled.evaluator $CONFIG_PATH
}

base_name="SamudrACE"

for year in $(seq -w 311 320); do
    run_eval "evaluator-config-yr${year}.yaml" \
             "${base_name}-arxiv-ckpt-ft-after-fto-yr${year}" \
             "01JY7H5WRR475Q5E6V2PA83SYQ"
done
