#!/bin/bash
#
# Adopted from the canonical reference launcher:
#   research/.claude/skills/launching-runs/run-train.reference.sh
# Only the BASELINE-SPECIFIC gantry block and the run_training calls at the
# bottom are edited; the GUARDRAILS block is copied verbatim (git grep the
# markers to detect drift).
#
# Prior launch waves for this baseline family (the residual-config sweep and
# the RH/qsat moisture eval) launched from earlier revisions of this script;
# see git history and the per-experiment records for their beaker/wandb links.
#
# Usage (run FROM this configs directory):
#   ./run-train.sh                  # launch every run_training call below
#   ./run-train.sh ft3step          # launch only calls matching a substring

set -euo pipefail

# === GUARDRAILS (copy verbatim from the reference; do not hand-edit) =========
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
  local N_GPUS="${3:-1}"
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"

  should_run "$config_filename" "$job_name" || { echo "skip (filter): $job_name"; return 0; }

  # path guard: the resolved local config must exist before we pay for gantry.
  # (cwd is REPO_ROOT here, so CONFIG_PATH is the repo-relative path.)
  if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: config not found: $REPO_ROOT/$CONFIG_PATH" >&2
    echo "       Check the filename and that you launched from the configs dir." >&2
    exit 1
  fi

  echo "launching: $job_name  ($CONFIG_PATH)"

  # --- BASELINE-SPECIFIC: edit only the block below for this baseline ---------
  # Validate locally to fail fast on config bugs before paying for GPU spin-up.
  python -m fme.ace.validate_config --config_type train "$CONFIG_PATH"

  # Extract additional gantry flags from "# arg: ..." headers in the YAML.
  local extra_args=()
  while IFS= read -r line; do
    [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
  done < "$CONFIG_PATH"

  gantry run \
    --name "$job_name" \
    --description 'Run ACE training (RH-input no-co2 humidity-trend)' \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace \
    --priority high \
    --cluster ai2/jupiter \
    --cluster ai2/titan \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP= \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --gpus "$N_GPUS" \
    --shared-memory "$((N_GPUS * 50))GiB" \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --allow-dirty \
    --system-python \
    --install "pip install --no-deps ." \
    "${extra_args[@]}" \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.ace.train "$CONFIG_PATH"
  # --- END BASELINE-SPECIFIC --------------------------------------------------
}

# =============================================================================
# Does a relative-humidity input improve humidity trends in a NO-CO2-INPUT model?
# Task:   research/tasks/2026-07-20-rh-input-noco2-humidity-trend.md
# Report: back-references ai2cm/reports rh-input-rollout-stability (PR #44)
#
# All arms drop the explicit CO2 input channel (global_mean_co2 removed from
# in_names + next_step_forcing_names; the long_46year_constant_co2 rollout is
# dropped, moot without a CO2 input). v2 no-residual, 4deg daily ERA5-only,
# fg16 sr0p125. Built on experiment/rh-multistep-finetune (the only branch with
# the saturation_normalization RH feature).
#
#   A  RH-input no-co2 base (single-step, 120 epochs). Launch now.
#   C  no-RH no-co2 3-step fine-tune (matched fine-tuned control): weights-only
#      init from the existing no-co2 no-RH base im4ecamc's best_ckpt.tar
#      (dataset 01KW0YE54G9NF9YWF000GVPMA8, mounted via the config "# arg:"
#      header). Launch now.
#   B  RH-input no-co2 3-step fine-tune of A (same ft recipe). DEFERRED: needs
#      A's checkpoint dataset; add its config + call and launch once A finishes.
#
# Baseline (single-step no-RH no-co2) is the pre-existing run im4ecamc.
# The clean matched RH-vs-no-RH comparison is B vs C (same branch code, both
# 3-step fine-tuned); A vs im4ecamc is single-step with a code-version caveat.
# =============================================================================

run_training "train-4deg-daily-v2-era5-only-fg16-sr0p125-no-residual-rh-input-append-noco2.yaml" "train-4deg-daily-v2-era5-only-fg16-sr0p125-no-residual-rh-input-append-noco2-rs0" 1
run_training "train-4deg-daily-v2-era5-only-no-residual-no-co2-ft3step.yaml"                     "train-4deg-daily-v2-era5-only-no-residual-no-co2-ft3step-rs0"                     1
