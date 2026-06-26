#!/bin/bash

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
REPO_ROOT=$(git rev-parse --show-toplevel)
# Set WANDB_USERNAME explicitly: the beaker job env does NOT inherit the shell
# value, so without this the run logs under the service-account key with a null
# username (unattributed). 'mcgibbon' is the correct human wandb identity; never
# 'jeremym' (the beaker account, which misattributes to the service account).
WANDB_USERNAME=mcgibbon

cd "$REPO_ROOT"

run_training() {
  local config_filename="$1"
  local job_name="$2"
  local N_GPUS="$3"
  local WORKSPACE="${4:-ai2/climate-titan}"
  local PRIORITY="${5:-urgent}"
  local CLUSTER="${6:-ai2/titan}"  # space-separated list allowed, e.g. "ai2/jupiter ai2/titan"
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"

  # Expand the (possibly multi-value) cluster list into repeated --cluster flags
  local cluster_args=()
  for c in $CLUSTER; do
    cluster_args+=(--cluster "$c")
  done

  python -m fme.ace.validate_config --config_type train "$CONFIG_PATH"

  # Extract additional args from config header
  local extra_args=()
  while IFS= read -r line; do
    [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
  done < "$CONFIG_PATH"

  gantry run \
    --name "$job_name" \
    --description 'Run ACE training (AIMIP-like baseline, 1°/daily)' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace "$WORKSPACE" \
    --priority "$PRIORITY" \
    --timeout 0 \
    --no-logs \
    "${cluster_args[@]}" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP= \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --gpus "$N_GPUS" \
    --shared-memory 400GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --allow-dirty \
    --system-python \
    --install "pip install --no-deps ." \
    "${extra_args[@]}" \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.ace.train "$CONFIG_PATH"
}

# =============================================================================
# 1°/daily ERA5-only baselines: 1° ports of the 4°/daily reference->v1->residual->v2
# baseline chain (task research/tasks/2026-06-23-rerun-4deg-daily-era5-baselines-at-1deg).
#
# Strict ports of the 4° configs (data -> 1°/daily ERA5 zarr + 1° stats; 60 epochs;
# train time_buffer 1 / no pool size; validation time_buffer 3; num_data_workers 8;
# inference forward_steps_in_memory 40; inference + ema_checkpoint epochs aligned to
# land on the final epoch). All launches: 4 GPUs, ai2/climate-titan, urgent,
# ai2/titan only (run_training defaults).
#
# The reference (#1, plain recipe) is intentionally omitted: an existing finished
# 1° run already covers it (wandb train-1deg-daily-era5-only-rs0 / vnhlcx31, 120 ep).
# =============================================================================

# --- v1 ERA5-only non-residual (shared T-norm + appended GM-T), seed 0 ---
# [already launched 2026-06-23] run_training "train-1deg-daily-v1-era5-only.yaml" "train-1deg-daily-v1-era5-only-rs0" 4

# --- v1 residual fg16xws64 (sr0.125), seed 0 ---
# [already launched 2026-06-23] run_training "train-1deg-daily-v1-era5-only-fg16-sr0p125-residual.yaml" "train-1deg-daily-v1-era5-only-fg16-sr0p125-residual-rs0" 4

# --- v2 ERA5-only residual (stitched train window), seed 0 ---
# [already launched 2026-06-23] run_training "train-1deg-daily-v2-era5-only.yaml" "train-1deg-daily-v2-era5-only-rs0" 4

# --- v2 ERA5-only NO-residual (stitched window, residual_prediction off), seed 0 ---
# 1deg-climate-improvement-transfer investigation: v2 recipe with residual prediction
# disabled (one-line flip from train-1deg-daily-v2-era5-only.yaml), fg16xws64 +
# stitched window otherwise held fixed. 1deg analog of the 4deg
# train-4deg-daily-v2-era5-only-no-residual run (znnaox7t).
run_training "train-1deg-daily-v2-era5-only-no-residual.yaml" "train-1deg-daily-v2-era5-only-no-residual-rs0" 4
