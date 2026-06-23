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
run_training "train-4deg-daily-v2-era5-only.yaml" "train-4deg-daily-v2-era5-only-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"

# =============================================================================
# fd-CRPS loss-weight perturbations off the v2 baseline (control = v2 default,
# c9d0e1 — crps 0.9 / energy 0.1, no fd-CRPS term). Identical to v2 except the
# loss kwargs: each re-introduces the finite-difference CRPS term (levels 2),
# and c5d5e0 also drops the energy score. Notation c·d·e = crps_weight /
# finite_difference_crps_weight / energy_score_weight. See
# research/investigations/2026-06-23-fd-crps-coherence-weighting.md.
# =============================================================================

# --- c7d2e1: crps 0.7 / fd-crps 0.2 / energy 0.1, fd levels 2 (1 GPU; Jupiter+Titan, high) ---
run_training "train-4deg-daily-v2-era5-only-c7d2e1.yaml" "train-4deg-daily-v2-era5-only-c7d2e1-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"

# --- c4d4e2: crps 0.4 / fd-crps 0.4 / energy 0.2, fd levels 2 (1 GPU; Jupiter+Titan, high) ---
run_training "train-4deg-daily-v2-era5-only-c4d4e2.yaml" "train-4deg-daily-v2-era5-only-c4d4e2-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"

# --- c5d4e1: crps 0.5 / fd-crps 0.4 / energy 0.1, fd levels 2 (1 GPU; Jupiter+Titan, high) ---
run_training "train-4deg-daily-v2-era5-only-c5d4e1.yaml" "train-4deg-daily-v2-era5-only-c5d4e1-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"

# --- c5d5e0: crps 0.5 / fd-crps 0.5 / energy 0.0, fd levels 2 (1 GPU; Jupiter+Titan, high) ---
run_training "train-4deg-daily-v2-era5-only-c5d5e0.yaml" "train-4deg-daily-v2-era5-only-c5d5e0-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"
