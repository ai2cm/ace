#!/bin/bash

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
 # since we use a service account API key for wandb, we use the beaker username to set the wandb username by default
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
WANDB_PROJECT=${WANDB_PROJECT:-TorchCompileTest}
BEAKER_WORKSPACE=${BEAKER_WORKSPACE:-ai2/ace}
BEAKER_CLUSTER=${BEAKER_CLUSTER:-"ai2/jupiter ai2/titan"}
BEAKER_PRIORITY=${BEAKER_PRIORITY:-normal}
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=2

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

run_training() {
  local config_filename="$1"
  local job_name="$2"
  local job_group="$3"
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"

  python -m fme.ace.validate_config --config_type train "$CONFIG_PATH"

  # Extract additional args from config header
  local extra_args=()
  while IFS= read -r line; do
    [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
  done < "$CONFIG_PATH"

  local cluster_args=()
  for cluster in $BEAKER_CLUSTER; do
    cluster_args+=(--cluster "$cluster")
  done

  gantry run \
    --name "$job_name" \
    --task-name "$job_name" \
    --description 'Run ACE2-ERA5 training' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace "$BEAKER_WORKSPACE" \
    --priority "$BEAKER_PRIORITY" \
    --min-runtime 4h \
    "${cluster_args[@]}" \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP="$job_group" \
    --env WANDB_PROJECT="$WANDB_PROJECT" \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --gpus $N_GPUS \
    --shared-memory 200GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --system-python \
    --allow-dirty \
    --install "pip install --no-deps ." \
    --allow-dirty \
    "${extra_args[@]}" \
    -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_PATH
}

# A/B: identical configs except `stepper.step.config.compile: true` in the second.
JOB_GROUP=${JOB_GROUP:-torch-compile-ab}
run_training "ace-train-config-4deg-AIMIP-nc-sfno.yaml" "torch-compile-ab-4deg-nc-sfno-eager" "$JOB_GROUP"
run_training "ace-train-config-4deg-AIMIP-nc-sfno-compile.yaml" "torch-compile-ab-4deg-nc-sfno-compile" "$JOB_GROUP"
