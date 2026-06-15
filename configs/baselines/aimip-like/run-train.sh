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
# Residual-config hyperparameter sweep (filter_num_groups x spectral_ratio)
# Task: research/tasks/2026-06-15-residual-config-hyperparameter-sweep.md
# Investigation: 2026-06-12-residual-recipe-selection
#
# Branch: experiment/2026-06-15-residual-config-sweep, cut from ace main
# (6bb72fa53) with NO extra code commits -- main already has residual
# prediction, spectral_ratio, clip_latent_global_means, and the
# persistence_names constant-CO2 inference loop. Inference therefore runs
# sequentially (the concurrent-inline-inference BatchedPredictor path is NOT
# pulled in; it is what crashed in the aimip-like Wave 18 reruns).
#
# All configs derive from the latest era5-only-residual config (the version
# with the 10year_insample + long_46year_constant_co2 eval loops), changed
# only by: max_epochs 60 -> 120, the swept knob(s), and seed.
#
# Grid (11 runs, 1 GPU each, embed_dim 512, ~120 epochs):
#   baseline (fg1, sr1.0):      seeds 0, 1, 2
#   spectral_ratio sweep:       sr 0.50, sr 0.25            (seed 0)
#   filter_num_groups sweep:    fg 4, fg 8, fg 16           (seed 0)
#   intersection:               fg8xsr0.25, fg8xsr0.50, fg4xsr0.25 (seed 0)
# Select on validation skill; accept if inference is not significantly worse.
# =============================================================================

# --- Wave 1: baseline residual, 3 seeds (Jupiter+Titan, high) ---
run_training "train-4deg-daily-v1-era5-only-residual.yaml"      "train-4deg-daily-v1-era5-only-residual-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"
run_training "train-4deg-daily-v1-era5-only-residual-rs1.yaml"  "train-4deg-daily-v1-era5-only-residual-rs1" 1 ai2/ace high "ai2/jupiter ai2/titan"
run_training "train-4deg-daily-v1-era5-only-residual-rs2.yaml"  "train-4deg-daily-v1-era5-only-residual-rs2" 1 ai2/ace high "ai2/jupiter ai2/titan"

# --- Wave 2: spectral_ratio sweep (seed 0) (Jupiter+Titan, high) ---
run_training "train-4deg-daily-v1-era5-only-sr0p50-residual.yaml" "train-4deg-daily-v1-era5-only-sr0p50-residual-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"
run_training "train-4deg-daily-v1-era5-only-sr0p25-residual.yaml" "train-4deg-daily-v1-era5-only-sr0p25-residual-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"

# --- Wave 3: filter_num_groups sweep (seed 0) (Jupiter+Titan, high) ---
run_training "train-4deg-daily-v1-era5-only-fg4-residual.yaml"  "train-4deg-daily-v1-era5-only-fg4-residual-rs0"  1 ai2/ace high "ai2/jupiter ai2/titan"
run_training "train-4deg-daily-v1-era5-only-fg8-residual.yaml"  "train-4deg-daily-v1-era5-only-fg8-residual-rs0"  1 ai2/ace high "ai2/jupiter ai2/titan"
run_training "train-4deg-daily-v1-era5-only-fg16-residual.yaml" "train-4deg-daily-v1-era5-only-fg16-residual-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"

# --- Wave 4: fg x sr intersection (seed 0) (Jupiter+Titan, high) ---
run_training "train-4deg-daily-v1-era5-only-fg8-sr0p25-residual.yaml" "train-4deg-daily-v1-era5-only-fg8-sr0p25-residual-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"
run_training "train-4deg-daily-v1-era5-only-fg8-sr0p50-residual.yaml" "train-4deg-daily-v1-era5-only-fg8-sr0p50-residual-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"
run_training "train-4deg-daily-v1-era5-only-fg4-sr0p25-residual.yaml" "train-4deg-daily-v1-era5-only-fg4-sr0p25-residual-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"
