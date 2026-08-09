#!/bin/bash

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
WANDB_USERNAME=${WANDB_USERNAME:-bhenn1983}
REPO_ROOT=$(git rev-parse --show-toplevel)
JOB_GROUP="ace2s-snowprog-daily"
CLUSTER="${CLUSTER:-ai2/jupiter}"

# train_loader.batch_size is a global batch split across ranks, so more ranks means
# less activation memory per GPU for the same math. 4 ranks fits titan's 180 GiB
# B200s; jupiter's 80 GiB H100s need 8.
case "$CLUSTER" in
  ai2/titan) N_GPUS=4 ;;
  ai2/jupiter) N_GPUS=8 ;;
  *) echo "no GPU-memory profile for cluster $CLUSTER" >&2; exit 1 ;;
esac

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

run_training() {
  local config_filename="$1"
  local job_name="$2"
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"
  shift 2

  local ckpt_dataset=""
  local job_group="$JOB_GROUP"
  local override_args=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --ckpt) ckpt_dataset="$2"; shift 2 ;;
      --group) job_group="$2"; shift 2 ;;
      *) override_args+=("$1"); shift ;;
    esac
  done

  local ckpt_arg=()
  if [[ -n "$ckpt_dataset" ]]; then
    ckpt_arg=(--dataset "$ckpt_dataset:/weights")
  fi

  python -m fme.ace.validate_config --config_type train "$CONFIG_PATH"

  # Extract additional args from config header
  local extra_args=()
  while IFS= read -r line; do
    [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
  done < "$CONFIG_PATH"

  gantry run \
    --name "$job_name" \
    --task-name "$job_name" \
    --description "ACE2S snow-transform daily training: $job_name" \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority high \
    --preemptible \
    --cluster "$CLUSTER" \
    --weka climate-default:/climate-default \
    "${extra_args[@]}" \
    "${ckpt_arg[@]}" \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP="$job_group" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --gpus $N_GPUS \
    --shared-memory 400GiB \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_PATH \
    ${override_args:+--override "${override_args[@]}"}
}

# 1-step pre-training only, compared against the naive-encoding baselines in
# ../2026-08-07-ace2s-snow-prognostic-daily (same W&B group).
# To run a subset, comment out the ones you don't want.
run_training "era5-snow-log1p-1-step-pretrain-daily.yaml"    "ace2s-snowprog-era5-daily-log1p-1-step-pretrain-rs0"    "seed=0"
run_training "era5-snow-quantile-1-step-pretrain-daily.yaml" "ace2s-snowprog-era5-daily-quantile-1-step-pretrain-rs0" "seed=0"
run_training "cm4-snow-log1p-1-step-pretrain-daily.yaml"     "ace2s-snowprog-cm4-daily-log1p-1-step-pretrain-rs0"     "seed=0"
run_training "cm4-snow-quantile-1-step-pretrain-daily.yaml"  "ace2s-snowprog-cm4-daily-quantile-1-step-pretrain-rs0"  "seed=0"
