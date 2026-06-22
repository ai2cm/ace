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
# Branch: experiment/2026-06-15-residual-config-sweep, cut from ace main --
# main already has residual prediction, spectral_ratio,
# clip_latent_global_means, and the persistence_names constant-CO2 inference
# loop. Merged on top: feature/concurrent-inline-inference (reconciled onto
# main's #1227 InferenceSummary interface) so inline inference batches
# concurrently, and fix/broadcast-ensemble-time-ordering (c54bc54cd), whose
# block-ordered broadcast_ensemble is required for concurrent inference with
# n_ensemble_per_ic > 1 (otherwise it crashes on a time-coordinate mismatch).
#
# All configs derive from the latest era5-only-residual config (the version
# with the 10year_insample + long_46year_constant_co2 eval loops), changed
# only by: max_epochs 60 -> 120, the swept knob(s), and seed.
#
# global_mean_co2 is in next_step_forcing_names (prescribed forcing, not
# prognostic) for every config below.
#
# Grid (12 runs, 1 GPU each, ~120 epochs; embed_dim 512 unless noted):
#   baseline (fg1, sr1.0):      seeds 0, 1, 2
#   spectral_ratio sweep:       sr 0.50, sr 0.25            (seed 0)
#   filter_num_groups sweep:    fg 4, fg 8, fg 16           (seed 0)
#   intersection:               fg8xsr0.25, fg8xsr0.50, fg4xsr0.25 (seed 0)
#   large-model + deep spectral cut: embed_dim 1024, sr 0.125 (seed 0)
# Select on validation skill; accept if inference is not significantly worse.
# =============================================================================

# --- LAUNCHED 2026-06-16 to ai2/ace from commit 77418ebd4 (12 jobs, 1 GPU each, jupiter+titan high). Lines commented to prevent re-submission; see experiment records for beaker/wandb links. ---
# --- Wave 1: baseline residual, 3 seeds (Jupiter+Titan, high) ---
# run_training "train-4deg-daily-v1-era5-only-residual.yaml"      "train-4deg-daily-v1-era5-only-residual-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"
# run_training "train-4deg-daily-v1-era5-only-residual-rs1.yaml"  "train-4deg-daily-v1-era5-only-residual-rs1" 1 ai2/ace high "ai2/jupiter ai2/titan"
# run_training "train-4deg-daily-v1-era5-only-residual-rs2.yaml"  "train-4deg-daily-v1-era5-only-residual-rs2" 1 ai2/ace high "ai2/jupiter ai2/titan"

# --- Wave 2: spectral_ratio sweep (seed 0) (Jupiter+Titan, high) ---
# run_training "train-4deg-daily-v1-era5-only-sr0p50-residual.yaml" "train-4deg-daily-v1-era5-only-sr0p50-residual-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"
# run_training "train-4deg-daily-v1-era5-only-sr0p25-residual.yaml" "train-4deg-daily-v1-era5-only-sr0p25-residual-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"

# --- Wave 3: filter_num_groups sweep (seed 0) (Jupiter+Titan, high) ---
# run_training "train-4deg-daily-v1-era5-only-fg4-residual.yaml"  "train-4deg-daily-v1-era5-only-fg4-residual-rs0"  1 ai2/ace high "ai2/jupiter ai2/titan"
# run_training "train-4deg-daily-v1-era5-only-fg8-residual.yaml"  "train-4deg-daily-v1-era5-only-fg8-residual-rs0"  1 ai2/ace high "ai2/jupiter ai2/titan"
# run_training "train-4deg-daily-v1-era5-only-fg16-residual.yaml" "train-4deg-daily-v1-era5-only-fg16-residual-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"

# --- Wave 4: fg x sr intersection (seed 0) (Jupiter+Titan, high) ---
# run_training "train-4deg-daily-v1-era5-only-fg8-sr0p25-residual.yaml" "train-4deg-daily-v1-era5-only-fg8-sr0p25-residual-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"
# run_training "train-4deg-daily-v1-era5-only-fg8-sr0p50-residual.yaml" "train-4deg-daily-v1-era5-only-fg8-sr0p50-residual-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"
# run_training "train-4deg-daily-v1-era5-only-fg4-sr0p25-residual.yaml" "train-4deg-daily-v1-era5-only-fg4-sr0p25-residual-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"

# --- Wave 5: embed_dim 1024 + spectral_ratio 0.125 (seed 0) (Jupiter+Titan, high) ---
# run_training "train-4deg-daily-v1-era5-only-n1024-sr0p125-residual.yaml" "train-4deg-daily-v1-era5-only-n1024-sr0p125-residual-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"

# --- Wave 6: complete the fg x sr grid (seed 0, embed_dim 512) (Jupiter+Titan, high) ---
# Fills the missing cells of the fg {4,8,16} x sr {0.5, 0.25, 0.125} grid at
# embed_dim 512. Already covered by Wave 4: fg4xsr0.25, fg8xsr0.25, fg8xsr0.50.
# (n1024xsr0.125 is embed_dim 1024, not part of this 512 grid.) 6 new runs.
# --- LAUNCHED 2026-06-17 to ai2/ace from commit f6632e8d5 (6 jobs, 1 GPU each, jupiter+titan high). Lines commented to prevent re-submission; see experiment records for beaker/wandb links. ---
# run_training "train-4deg-daily-v1-era5-only-fg4-sr0p50-residual.yaml"   "train-4deg-daily-v1-era5-only-fg4-sr0p50-residual-rs0"   1 ai2/ace high "ai2/jupiter ai2/titan"
# run_training "train-4deg-daily-v1-era5-only-fg4-sr0p125-residual.yaml"  "train-4deg-daily-v1-era5-only-fg4-sr0p125-residual-rs0"  1 ai2/ace high "ai2/jupiter ai2/titan"
# run_training "train-4deg-daily-v1-era5-only-fg8-sr0p125-residual.yaml"  "train-4deg-daily-v1-era5-only-fg8-sr0p125-residual-rs0"  1 ai2/ace high "ai2/jupiter ai2/titan"
# run_training "train-4deg-daily-v1-era5-only-fg16-sr0p50-residual.yaml"  "train-4deg-daily-v1-era5-only-fg16-sr0p50-residual-rs0"  1 ai2/ace high "ai2/jupiter ai2/titan"
# run_training "train-4deg-daily-v1-era5-only-fg16-sr0p25-residual.yaml"  "train-4deg-daily-v1-era5-only-fg16-sr0p25-residual-rs0"  1 ai2/ace high "ai2/jupiter ai2/titan"
# run_training "train-4deg-daily-v1-era5-only-fg16-sr0p125-residual.yaml" "train-4deg-daily-v1-era5-only-fg16-sr0p125-residual-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"

# --- Wave 7: extend the grid into deeper spectral cuts (seed 0, embed_dim 512) (Jupiter+Titan, high) ---
# Intersect filter_num_groups {8,16,32,64} with the two deepest spectral cuts:
# spectral_ratio 0.125 -> 64 spectral channels (ws64) and 0.0625 -> 32 (ws32),
# at embed_dim 512. Run names use the wsNN convention (latent spectral width)
# instead of srNN. filter_num_groups must divide the channel count, so fg64xws32
# (64 > 32) is invalid and omitted. Already covered by Wave 6: fg8xws64 and
# fg16xws64 (the fg8/fg16-sr0p125 runs). 5 new runs.
# --- LAUNCHED 2026-06-20 to ai2/ace from commit e334b4be1 (5 jobs, 1 GPU each, jupiter+titan high). Lines commented to prevent re-submission; see experiment records for beaker/wandb links. ---
# run_training "train-4deg-daily-v1-era5-only-fg32-ws64-residual.yaml" "train-4deg-daily-v1-era5-only-fg32-ws64-residual-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"
# run_training "train-4deg-daily-v1-era5-only-fg64-ws64-residual.yaml" "train-4deg-daily-v1-era5-only-fg64-ws64-residual-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"
# run_training "train-4deg-daily-v1-era5-only-fg8-ws32-residual.yaml"  "train-4deg-daily-v1-era5-only-fg8-ws32-residual-rs0"  1 ai2/ace high "ai2/jupiter ai2/titan"
# run_training "train-4deg-daily-v1-era5-only-fg16-ws32-residual.yaml" "train-4deg-daily-v1-era5-only-fg16-ws32-residual-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"
# run_training "train-4deg-daily-v1-era5-only-fg32-ws32-residual.yaml" "train-4deg-daily-v1-era5-only-fg32-ws32-residual-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"

# =============================================================================
# Pre-1979 ERA5 stability test (seed 0, embed_dim 512, fg16 x sr0.125 base)
# Task: research/tasks/2026-06-16-test-whether-including-pre-1979-era5-lets-a-residual-model-train-stably.md
# Goal: 2026-06-11-stable-long-rollouts
#
# The selected stable residual base (fg16 x spectral_ratio 0.125, "ws64",
# embed_dim 512; run j8r0z322) trains on 1979-1993 + 1995-2013. This config is
# identical except the train_loader prepends the pre-1979 back-extension
# (1940-01-01 -> 1978-12-31) as a third concat subset -- a clean A/B on the
# data range. Validation (1994 + 2014) and all inference eval loops are
# unchanged so val skill and the 46-year rollout are directly comparable to the
# base. No stitch-skip surgery: the diff diagnostic on this daily 4deg dataset
# shows pre-1979 timestep jumps <= the un-skipped 1979-2013 baseline and no jump
# at the documented 1946 stitches, so any instability is attributable to the
# data range itself, not an unskipped stitch.
# =============================================================================
run_training "train-4deg-daily-v1-era5-only-pre1979-fg16-sr0p125-residual.yaml" "train-4deg-daily-v1-era5-only-pre1979-fg16-sr0p125-residual-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"
