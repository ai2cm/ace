#!/bin/bash
#
# SHiELD+ -> ERA5 fine-tune launcher, family-1 variant C: exclude the SFNO
# global MLP. Separate from run-train.sh because a fine-tune mounts the donor
# checkpoint (--dataset ... :/ckpt.tar); the guardrails block is verbatim.
#
# research: tasks/2026-07-08-finetune-era5-excluding-sfno-global-mlp.md
#           goal   2026-07-08-transfer-learning-forcing-response-verdict
#
# Usage (run FROM this configs directory):
#   ./run-train-ft-exclude-global-mlp.sh            # launch both arms
#   ./run-train-ft-exclude-global-mlp.sh lr1e-5     # only the lr1e-5 arm

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
  echo "       Invoke this script FROM its own configs directory, not the repo root." >&2
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

# --- Fine-tune donor checkpoint (SHiELD+ modern 1-step pre-train, epoch 120) --
# The donor is the modern-arm pre-train (research experiment
# 2026-07-13-transfer-learning-modern-pretrain-rs0; beaker
# 01KXGPGCRQZVVM26N8A9ZX7MZ5 / wandb jnlquua6). We mount the final-epoch EMA
# stepper checkpoint (ema_ckpt_0120.tar) at /ckpt.tar; the config loads it
# weights-only via stepper_training.parameter_init.weights_path (fresh optimizer).
# ema_ckpt_0120 is the epoch-120 EMA model — the response-bearing model verified
# by the pre-train's inline +2K/+4K inference (validate_using_ema: true).
DONOR_DATASET="01KXGPGD058C13PYAQG47D6XY1"   # result dataset of beaker 01KXGPGC…
DONOR_CKPT="training_checkpoints/ema_ckpt_0120.tar"

cd "$REPO_ROOT"

run_training() {
  local config_filename="$1"
  local job_name="$2"
  local N_GPUS="${3:-1}"
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"

  should_run "$config_filename" "$job_name" || { echo "skip (filter): $job_name"; return 0; }

  # path guard: the resolved local config must exist before we pay for gantry.
  if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: config not found: $REPO_ROOT/$CONFIG_PATH" >&2
    echo "       Check the filename and that you launched from the configs dir." >&2
    exit 1
  fi

  echo "launching: $job_name  ($CONFIG_PATH)"

  # --- BASELINE-SPECIFIC: fine-tune mounts the donor checkpoint at /ckpt.tar ---
  python -m fme.ace.validate_config --config_type train "$CONFIG_PATH"

  local extra_args=()
  while IFS= read -r line; do
    [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
  done < "$CONFIG_PATH"

  gantry run \
    --name "$job_name" \
    --description 'SHiELD+ -> ERA5 fine-tune, exclude SFNO global MLP (family-1 variant C)' \
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
    --dataset "$DONOR_DATASET:$DONOR_CKPT:/ckpt.tar" \
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
# Two arms, differing ONLY in learning rate. Both load the modern SHiELD+ donor
# and fine-tune on 4°/daily ERA5 with filter_preserves_global_mean: true
# (ace #1358, merged into this experiment branch), so the fine-tune cannot
# relearn ERA5's global means through the l=0 per-mode "global MLP". Model
# architecture / channel order / corrector / tnorm / residual_prediction:false
# are verbatim from the donor; only data, normalization stats and inline
# inference are ERA5. Recipe = modern v2 (1-step, EnsembleLoss, max_epochs 120).
# Post-hoc +4K evaluation: run-inference-ft-exclude-global-mlp.sh once trained.
# =============================================================================

# --- variant C, lr 1e-4 (modern recipe lr), seed 0 (1 GPU; Jupiter+Titan, high) ---
run_training "train-4deg-daily-ft-era5-exclude-global-mlp-lr1e-4.yaml" "ft-4deg-daily-era5-exclude-global-mlp-lr1e-4-rs0" 1

# --- variant C, lr 1e-5 (gentler fine-tune), seed 0 (1 GPU; Jupiter+Titan, high) ---
run_training "train-4deg-daily-ft-era5-exclude-global-mlp-lr1e-5.yaml" "ft-4deg-daily-era5-exclude-global-mlp-lr1e-5-rs0" 1
