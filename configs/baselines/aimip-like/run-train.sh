#!/bin/bash

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
REPO_ROOT=$(git rev-parse --show-toplevel)
# NB: WANDB_USERNAME must be set to the human owner (mcgibbon), NOT the beaker
# account (jeremym) — the SA API key carries no username, so without this the run
# is attributed to nobody, and setting it to the beaker account misattributes it
# to the service account. It is set explicitly on the gantry --env below.
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
    --description 'Run ACE training (AIMIP-like baseline)' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace "$WORKSPACE" \
    --priority "$PRIORITY" \
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
# Already launched and running as wandb oshj5u79 / beaker 01KVTHCVW0DB3F4Q8CNTV8ZRB7
# (relaunched 2026-06-23 @ 6e8cf916f). Commented out so this script does not
# relaunch it; uncomment to launch a fresh v2 seed.
# run_training "train-4deg-daily-v2-era5-only.yaml" "train-4deg-daily-v2-era5-only-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"

# =============================================================================
# v2 ERA5-only residual baseline + total-energy-budget corrector (challenger).
#
# Identical to the v2 baseline recipe except the atmosphere corrector adds
# total_energy_budget_correction (method: constant_temperature,
# constant_unaccounted_heating: 0.0). Evaluates whether enabling the energy
# corrector improves the v2 4°/daily baseline; if it pans out it forks a v2
# successor. unaccounted_heating is 0 per the current-runs convention (the 6.62
# W/m**2 ERA5 1979-2008 residual is NOT applied here).
# =============================================================================

# --- v2 + energy corrector, seed 0 (1 GPU; Jupiter+Titan, high) ---
# Already launched (wandb-tracked; beaker 01KVTX5W42793TVEWC8P19A4E9). Commented
# out so this script does not relaunch it; uncomment to launch a fresh seed.
# run_training "train-4deg-daily-v2-era5-only-energy-corrector.yaml" "train-4deg-daily-v2-era5-only-energy-corrector-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"

# =============================================================================
# v2 ERA5-only residual baseline + total-energy-budget corrector WITH
# unaccounted heating (challenger sibling).
#
# Identical to the energy-corrector challenger above except
# constant_unaccounted_heating is set to 6.62 W/m**2 (the ERA5 1979-2008
# column-integrated energy residual) instead of 0.0. Pairs with the
# unaccounted_heating=0 run to isolate the effect of applying the residual
# heating term under the constant_temperature total-energy budget correction.
# =============================================================================

# --- v2 + energy corrector + unaccounted heating 6.62, seed 0 (1 GPU; Jupiter+Titan, high) ---
# Already launched (wandb w2ilktx7; beaker 01KVX7VP9NQ4PKS86K8QH8N3JM). Commented
# out so this script does not relaunch it; uncomment to launch a fresh seed.
# run_training "train-4deg-daily-v2-era5-only-energy-corrector-uh6p62.yaml" "train-4deg-daily-v2-era5-only-energy-corrector-uh6p62-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"

# =============================================================================
# v2 ERA5-only residual baseline + total-energy-budget corrector EXCLUDING the
# top vertical level from the temperature correction (challenger).
#
# Identical to the energy-corrector challenger (heating-0) except the
# total_energy_budget_correction sets exclude_top_levels: 1, so the uniform
# temperature addition is applied to levels 1-7 only (level 0, the model top, is
# excluded). The global magnitude of the correction is still computed across all
# levels. Tests whether excluding the top level removes the positive
# air_temperature_0 bias the all-levels corrector introduced (vs v2 control
# oshj5u79) without reintroducing energy-budget drift.
# =============================================================================

# --- v2 + energy corrector, exclude top level, seed 0 (1 GPU; Jupiter+Titan, high) ---
# Already launched (wandb bre6554p; beaker 01KW2D1V2F4FY85R3N7M0GFWEV). Commented
# out so this script does not relaunch it; uncomment to launch a fresh seed.
# run_training "train-4deg-daily-v2-era5-only-energy-corrector-excl-top1.yaml" "train-4deg-daily-v2-era5-only-energy-corrector-excl-top1-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"

# =============================================================================
# v2 ERA5-only residual baseline + total-energy-budget corrector EXCLUDING the
# top TWO vertical levels from the temperature correction (challenger).
#
# Identical to the excl-top1 arm above except the
# total_energy_budget_correction sets exclude_top_levels: 2, so the uniform
# temperature addition is applied to levels 2-7 only (levels 0 and 1 are
# excluded). The global magnitude of the correction is still computed across all
# levels. Follow-on to excl-top1: that arm only halved the air_temperature_0
# warm bias and relocated the warm load onto level 1, so this arm spares level 1
# as well to test whether the bias clears without reintroducing energy-budget
# drift. See research investigation
# 2026-06-30-energy-corrector-exclude-top-2-levels.
# =============================================================================

# --- v2 + energy corrector, exclude top 2 levels, seed 0 (1 GPU; Jupiter+Titan, high) ---
# Belongs to a different investigation and its config is not on this branch;
# commented out so this launcher only launches the pressure-winds arm below.
# run_training "train-4deg-daily-v2-era5-only-energy-corrector-excl-top2.yaml" "train-4deg-daily-v2-era5-only-energy-corrector-excl-top2-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"

# =============================================================================
# v2 ERA5-only residual baseline + pressure-surface winds & geostrophic-imbalance
# diagnostics (this branch's experiment).
#
# Identical recipe to the v2 baseline except: out_names adds UGRD/VGRD at
# 200/500/850 hPa and h200/h850 as diagnostic-only outputs, and every inference
# case enables mean_denorm (for the ageostrophic_wind_speed_{lvl} /
# ageostrophic_speed_residual_{lvl} derived variables) and the wind_consistency
# companion aggregator. Trains a model that predicts the pressure-surface winds
# and evaluates it with the geostrophic-imbalance aggregator (incl. out-of-sample
# and the 46-year rollout), to decide whether to adopt these fields + the
# aggregator. See research task
# 2026-06-22-add-pressure-surface-winds-for-geostrophic-diagnostics.
# =============================================================================

# --- v2 + pressure-surface winds, seed 0 (1 GPU; Jupiter+Titan, high) ---
run_training "train-4deg-daily-v2-era5-only-pressure-winds.yaml" "train-4deg-daily-v2-era5-only-pressure-winds-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"
