#!/bin/bash

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=8
PRETRAIN_DATASET="01JVQYZB8N5C37RN564Q8P24AP"

cd "$REPO_ROOT"

run_training() {
  local config_filename="$1"
  local job_name="$2"
  local stats_dataset="$3"
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"

  python -m fme.ace.validate_config --config_type train "$CONFIG_PATH"

  gantry run \
    --name "$job_name" \
    --description 'Run ACE training' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority normal \
    --preemptible \
    --cluster ai2/ceres-cirrascale \
    --cluster ai2/titan-cirrascale \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP= \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env TORCH_DISTRIBUTED_DEBUG=DETAIL \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "$stats_dataset:/statsdata" \
    --dataset $PRETRAIN_DATASET:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar \
    --gpus "$N_GPUS" \
    --shared-memory 400GiB \
    --weka climate-default:/climate-default \
    --budget ai2/climate \
    --no-conda \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.ace.train "$CONFIG_PATH"
}

base_name="train-camulator"

stats_1deg_dataset="andrep/2024-07-24-vertically-resolved-c96-1deg-shield-amip-ensemble-dataset-stats"

# To run a subset of these, comment out the ones you don't want to run
# and if needed use `--allow-dirty` in the gantry run command above.
# run_training "train-baseline.yaml" "$base_name-baseline" "$stats_1deg_dataset"
run_training "train-baseline_amip_shield_finetune.yaml" "$base_name-baseline-1deg-amip-shield-no-co2-finetune" "$stats_1deg_dataset"
