#!/bin/bash
#
# Training launcher for the ACE2S paper ablation suite (40-epoch pretrains
# off the frozen paper recipe; the 3-step detached fine-tunes follow as each
# pretrain finishes).
# Adopted from research/.claude/skills/launching-runs/run-train.reference.sh
#
# Usage (run FROM configs/baselines/era5-ace2s-ablations/):
#   ./run-train.sh                  # launch every arm below
#   ./run-train.sh rs1 rs2          # launch only the matching arms
#
# Suite-wide submission choices (Jeremy, 2026-09-23): beaker priority
# "normal" and NO CM_PRIORITY label, so the priority balancer leaves these
# jobs alone. 1-degree arms run on 8 GPUs of a single cluster.

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
  local CLUSTER="${4:-ai2/jupiter}"   # 1° runs: exactly one cluster
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

  # This suite deliberately sets no CM_PRIORITY label (the reference launcher's
  # default is dropped): the jobs run at beaker priority "normal", unmanaged by
  # the priority balancer.
  gantry run \
    --name "$job_name" \
    --description 'Run ACE training' \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace \
    --priority normal \
    --min-runtime 8h \
    --cluster "$CLUSTER" \
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

# Seed replicates of the paper recipe's pretrain (gjsqlvsf is seed 0).
run_training "ace2s-pretrain-rs1.yaml" "1deg-daily-ace2s-pretrain-rs1" 8
run_training "ace2s-pretrain-rs2.yaml" "1deg-daily-ace2s-pretrain-rs2" 8
run_training "ace2s-pretrain-rs3.yaml" "1deg-daily-ace2s-pretrain-rs3" 8

# One-knob ablations off the paper recipe's pretrain (seed 0).
run_training "ace2s-pretrain-deterministic.yaml" "1deg-daily-ace2s-det-pretrain-rs0" 8
run_training "ace2s-pretrain-crps-only.yaml" "1deg-daily-ace2s-crps-only-pretrain-rs0" 8
run_training "ace2s-pretrain-no-bottleneck.yaml" "1deg-daily-ace2s-no-bottleneck-pretrain-rs0" 8
run_training "ace2s-pretrain-6hourly.yaml" "1deg-6h-ace2s-pretrain-rs0" 8
# Resume of the 6-hourly arm after its epoch-10 NCCL-timeout failure (same
# wandb name so the run keeps its id; beaker suffixes the experiment name).
run_training "ace2s-pretrain-6hourly-resume.yaml" "1deg-6h-ace2s-pretrain-rs0" 8

# 4-degree arm (one GPU; needs the regenerated 2026-09-08 4-degree daily store on weka)
run_training "ace2s-pretrain-4deg.yaml" "4deg-daily-ace2s-pretrain-rs0" 1

# 4-degree arm, stage 2: 3-step detached fine-tune of the 4-degree pretrain
run_training "ace2s-finetune-4deg.yaml" "4deg-daily-ace2s-ft3-detached-rs0" 1

# 4-degree arm, seed replicates 1-3 (one GPU each)
run_training "ace2s-pretrain-4deg-rs1.yaml" "4deg-daily-ace2s-pretrain-rs1" 1
run_training "ace2s-pretrain-4deg-rs2.yaml" "4deg-daily-ace2s-pretrain-rs2" 1
run_training "ace2s-pretrain-4deg-rs3.yaml" "4deg-daily-ace2s-pretrain-rs3" 1
