#!/bin/bash
# A/B for keep_gradient_through_clamps on the v1-era5-only config (ensemble
# CRPS loss). 2 jobs, seed 0: baseline (flag off) vs STE (flag on), 1-GPU 4deg.
# Same-seed pair is bit-identical at init (STE leaves forward values unchanged)
# and diverges only through the corrector clamp gradient path. dry_fraction on
# PRATEsfc is logged by the 4 non-ensemble climate inference evals.
#
# Derived from configs/baselines/aimip-like/train-4deg-daily-v1-era5-only.yaml
# (the Wave 7 v1-era5-only baseline). Stats + data come from the weka
# climate-default mount; no separate stats dataset is needed.

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to repo root
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=1
RUN_GROUP="keep-gradient-clamps-v1-2026-06-19"

cd "$REPO_ROOT"

run_training() {
  local config_filename="$1"
  local job_name="$2"
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"

  python -m fme.ace.validate_config --config_type train "$CONFIG_PATH"

  gantry run \
    --name "$job_name" \
    --description 'A/B keep_gradient_through_clamps on v1-era5-only (ensemble), 1-GPU 4deg' \
    --yes \
    --timeout 0 \
    --beaker-image "$(cat "$REPO_ROOT"/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority high \
    --cluster ai2/jupiter \
    --cluster ai2/titan \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP="$RUN_GROUP" \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --gpus "$N_GPUS" \
    --shared-memory 400GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.ace.train "$CONFIG_PATH"
}

run_training "train-baseline.yaml" "kgc-v1-baseline-rs0"
run_training "train-ste.yaml"      "kgc-v1-ste-rs0"
