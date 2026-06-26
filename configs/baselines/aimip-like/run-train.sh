#!/bin/bash

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
REPO_ROOT=$(git rev-parse --show-toplevel)

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
    --description 'Run ACE training (AIMIP-like baseline)' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace "$WORKSPACE" \
    --priority "$PRIORITY" \
    "${cluster_args[@]}" \
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
    --budget ai2/atec-climate \
    --allow-dirty \
    --system-python \
    --install "pip install --no-deps ." \
    "${extra_args[@]}" \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.ace.train "$CONFIG_PATH"
}

# =============================================================================
# v2 ERA5-only residual baseline (canonical).
#
# The selected 4°/daily residual recipe (filter_num_groups 16 × spectral_ratio
# 0.125 "ws64", embed_dim 512; residual-recipe-selection, run j8r0z322), with
# one change from v1: the train_loader splits each ERA5 production stream at the
# canonical in-window stitch boundaries (1986-04-01, 1993-08-01, 2000-01-01,
# 2010-01-01) so no training sample's 1-step residual target straddles a
# stream-to-stream discontinuity. See
# research/knowledge/era5-stitching-discontinuities.md. All weights-affecting
# settings other than the training data window are identical to v1.
# =============================================================================

# --- v2 baseline, seed 0 (1 GPU; Jupiter+Titan, high) ---
# Already launched (wandb rs6b7nyr); commented here so this branch's pre-1979
# variant launches without re-submitting the baseline.
# run_training "train-4deg-daily-v2-era5-only.yaml" "train-4deg-daily-v2-era5-only-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"

# =============================================================================
# Pre-1979 ERA5 stability test on the v2 stitched baseline (seed 0).
# Task: research/tasks/2026-06-16-test-whether-including-pre-1979-era5-lets-a-residual-model-train-stably.md
# Goal: 2026-06-11-stable-long-rollouts
#
# v2 baseline (1979-2013, stitched) with the pre-1979 back-extension (1940-1978)
# prepended to the train_loader, split into ~5-year streams at the stitch
# boundaries the global-mean 1-step finite-difference diagnostic found
# (specific_total_water_0 steps on Jan-1 of 1944/1949/1954/1959/1964/1969/1974).
# Validation (1994 + 2014) and all inference eval loops are identical to v2 for a
# clean A/B on the data range vs the v2 baseline (rs6b7nyr). seed 0, ~120 epochs.
# =============================================================================
# Already launched (wandb lbk553vf); commented so this branch launches only the no-residual variant.
# run_training "train-4deg-daily-v2-era5-only-pre1979.yaml" "train-4deg-daily-v2-era5-only-pre1979-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"

# =============================================================================
# No-residual control for the pre-1979 stability test (seed 0).
# Task: research/tasks/2026-06-26-add-a-no-residual-pre-1979-control-to-attribute-the-residual-instability.md
# Investigation: research/investigations/2026-06-22-pre1979-era5-residual-stability.md
#
# Identical to train-4deg-daily-v2-era5-only-pre1979.yaml (the residual pre-1979
# run, wandb lbk553vf) EXCEPT residual_prediction: true -> false. Isolates
# whether the late-training rollout-skill degradation seen on the residual
# pre-1979 run is residual-specific or a property of the pre-1979 data / longer
# training. Baseline: no-residual v2 (1979-on), wandb znnaox7t. seed 0, 120 epochs.
# =============================================================================
run_training "train-4deg-daily-v2-era5-only-no-residual-pre1979.yaml" "train-4deg-daily-v2-era5-only-no-residual-pre1979-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"
