#!/bin/bash

set -e

# === GUARDRAILS (copy verbatim from the reference; do not hand-edit) =========
# Source: research/.claude/skills/launching-runs/run-train.reference.sh
WANDB_IDENTITY="mcgibbon"   # the wandb username every run must attribute to

SCRIPT_PATH=$(git rev-parse --show-prefix)   # repo-root-relative dir of this script
REPO_ROOT=$(git rev-parse --show-toplevel)
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')

# WANDB attribution guard. The beaker job env does not carry WANDB_USERNAME, and
# the beaker account (jeremym) makes wandb fall back to the API-key service
# account, so an unset/null/jeremym value silently misattributes the run. Beaker
# specs are immutable, so a miss costs a full stop+relaunch+rewrite-every-record
# cycle — fail loud, before submit.
WANDB_USERNAME=${WANDB_USERNAME:-$WANDB_IDENTITY}
if [[ "$WANDB_USERNAME" != "$WANDB_IDENTITY" ]]; then
  echo "ERROR: WANDB_USERNAME='$WANDB_USERNAME' but runs must attribute to '$WANDB_IDENTITY'." >&2
  echo "       (BEAKER_USERNAME='$BEAKER_USERNAME' would misattribute to the wandb service account.)" >&2
  echo "       Run:  export WANDB_USERNAME=$WANDB_IDENTITY   before launching." >&2
  exit 1
fi

# cwd / path guard. An empty SCRIPT_PATH means the script was run from the repo
# root (or outside the configs dir): CONFIG_PATH would become "/<config>.yaml"
# and gantry would submit a doomed job even after local validate_config fails.
if [[ -z "$SCRIPT_PATH" ]]; then
  echo "ERROR: SCRIPT_PATH (git rev-parse --show-prefix) is empty." >&2
  echo "       Invoke run-train.sh FROM its own configs directory, not the repo root." >&2
  exit 1
fi

# Config-line filter. With no args every run_training call runs; with args, only
# calls whose config filename OR job name contains one of the substrings.
LAUNCH_FILTERS=("$@")
should_run() {  # should_run <config_filename> <job_name>
  [[ ${#LAUNCH_FILTERS[@]} -eq 0 ]] && return 0
  local f
  for f in "${LAUNCH_FILTERS[@]}"; do
    [[ "$1" == *"$f"* || "$2" == *"$f"* ]] && return 0
  done
  return 1
}

# Post-launch attribution assertion. gantry submits asynchronously, so the wandb
# run may not exist at submit time; call this once the run has registered (or as
# a standalone follow-up check) to confirm wandb really recorded it under
# WANDB_IDENTITY before you write records / move on.
#   assert_wandb_attribution <wandb_run_id> [wandb_project]   # default ai2cm/ace
assert_wandb_attribution() {
  local run_id="$1" project="${2:-ai2cm/ace}"
  python - "$run_id" "$project" "$WANDB_IDENTITY" <<'PY'
import sys
import wandb
run_id, project, expected = sys.argv[1], sys.argv[2], sys.argv[3]
got = wandb.Api().run(f"{project}/{run_id}").user.username
assert got == expected, f"wandb run {run_id} attributed to {got!r}, expected {expected!r}"
print(f"OK: wandb run {run_id} attributed to {got}")
PY
}
# === END GUARDRAILS =========================================================

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

  should_run "$config_filename" "$job_name" || { echo "skip (filter): $job_name"; return 0; }

  # path guard: the resolved local config must exist before we pay for gantry.
  if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: config not found: $REPO_ROOT/$CONFIG_PATH" >&2
    echo "       Check the filename and that you launched from the configs dir." >&2
    exit 1
  fi

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
run_training "train-4deg-daily-v2-era5-only-energy-corrector-excl-top2.yaml" "train-4deg-daily-v2-era5-only-energy-corrector-excl-top2-rs0" 1 ai2/ace high "ai2/jupiter ai2/titan"
