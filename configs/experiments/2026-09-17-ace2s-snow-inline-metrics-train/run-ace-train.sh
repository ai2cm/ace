#!/bin/bash

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
WANDB_USERNAME=${WANDB_USERNAME:-bhenn1983}
REPO_ROOT=$(git rev-parse --show-toplevel)
JOB_GROUP="ace2s-snow-inline-metrics"
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
    --description "ACE2S snow inline-metrics training: $job_name" \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority high \
    --min-runtime 8h \
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

# 1-step pre-training of the control and masked-naive arms on both datasets, with
# the anomaly_memory and snow_season aggregators in the inline inference.
# To run a subset, comment out the ones you don't want.
# The controls finished 2026-09-21 and are not relaunched. The masked-naive arms are
# relaunched on the per-land-area snow channels (see README), under new job names so
# the first runs, trained on the per-cell-area ERA5 data, keep theirs.
# run_training "cm4-control-1-step-pretrain-daily.yaml"       "ace2s-snowmetrics-cm4-daily-control-1-step-pretrain-rs0"       "seed=0"
run_training "cm4-masked-naive-1-step-pretrain-daily.yaml"  "ace2s-snowmetrics-cm4-daily-masked-naive-land-snow-1-step-pretrain-rs0"  "seed=0"
# run_training "era5-control-1-step-pretrain-daily.yaml"      "ace2s-snowmetrics-era5-daily-control-1-step-pretrain-rs0"      "seed=0"
run_training "era5-masked-naive-1-step-pretrain-daily.yaml" "ace2s-snowmetrics-era5-daily-masked-naive-land-snow-1-step-pretrain-rs0" "seed=0"
