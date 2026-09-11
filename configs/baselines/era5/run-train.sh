#!/bin/bash
#
# Training launcher for the no-energy-corrector + *_mean outputs model.
# Adopted from research/.claude/skills/launching-runs/run-train.reference.sh
#
# Usage (run FROM configs/baselines/era5/):
#   ./run-train.sh                  # launch the pretrain
#   ./run-train.sh pretrain         # same (filter match)
#   ./run-train.sh ft3              # launch only the fine-tune (needs /weights)

set -euo pipefail

# === GUARDRAILS (copy verbatim from the reference; do not hand-edit) =========
WANDB_IDENTITY="mcgibbon"

SCRIPT_PATH=$(git rev-parse --show-prefix)
REPO_ROOT=$(git rev-parse --show-toplevel)
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')

WANDB_USERNAME=${WANDB_USERNAME:-$WANDB_IDENTITY}
if [[ "$WANDB_USERNAME" != "$WANDB_IDENTITY" ]]; then
  echo "ERROR: WANDB_USERNAME='$WANDB_USERNAME' but runs must attribute to '$WANDB_IDENTITY'." >&2
  echo "       (BEAKER_USERNAME='$BEAKER_USERNAME' would misattribute to the wandb service account.)" >&2
  echo "       Run:  export WANDB_USERNAME=$WANDB_IDENTITY   before launching." >&2
  exit 1
fi

if [[ -z "$SCRIPT_PATH" ]]; then
  echo "ERROR: SCRIPT_PATH (git rev-parse --show-prefix) is empty." >&2
  echo "       Invoke run-train.sh FROM its own configs directory, not the repo root." >&2
  exit 1
fi

LAUNCH_FILTERS=("$@")
should_run() {
  [[ ${#LAUNCH_FILTERS[@]} -eq 0 ]] && return 0
  local f
  for f in "${LAUNCH_FILTERS[@]}"; do
    [[ "$1" == *"$f"* || "$2" == *"$f"* ]] && return 0
  done
  return 1
}

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
  local CLUSTER="${4:-ai2/jupiter}"   # 1° runs: exactly one cluster (jupiter@8, or titan@8 for BPTT)
  local PRIORITY="${5:-high}"        # beaker priority; pair a low arm with a "# arg: --env CM_PRIORITY=low" header
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"

  should_run "$config_filename" "$job_name" || { echo "skip (filter): $job_name"; return 0; }

  if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: config not found: $REPO_ROOT/$CONFIG_PATH" >&2
    echo "       Check the filename and that you launched from the configs dir." >&2
    exit 1
  fi

  echo "launching: $job_name  ($CONFIG_PATH)"

  python -m fme.ace.validate_config --config_type train "$CONFIG_PATH"

  local extra_args=()
  while IFS= read -r line; do
    [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
  done < "$CONFIG_PATH"

  local cm_priority_args=(--env CM_PRIORITY=high) arg prev=""
  for arg in "${extra_args[@]}"; do
    if [[ ("$prev" == --env && "$arg" == CM_PRIORITY=*) || "$arg" == --env=CM_PRIORITY=* ]]; then
      cm_priority_args=()
      break
    fi
    prev="$arg"
  done

  gantry run \
    --name "$job_name" \
    --description 'Run ACE training' \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace \
    --priority "$PRIORITY" \
    --min-runtime 8h \
    --cluster "$CLUSTER" \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=training \
    "${cm_priority_args[@]}" \
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
}

# Pretrain: 1-step, 40 epochs, 8 GPUs on jupiter
run_training \
  "ace-train-config-1-step-pretrain-daily-fg16-sr0p125-no-corr-mean.yaml" \
  "1deg-daily-no-corr-mean-pretrain-rs0" \
  8

# Fine-tune: 3-step BPTT (full backprop through rollout), 40 epochs. Multi-step
# BPTT at 1° goes to titan with 8 GPUs (OOM on titan@4; jupiter@8 holds less
# total memory than titan@4).
run_training \
  "ace-train-config-ft3-bptt-daily-fg16-sr0p125-no-corr-mean.yaml" \
  "1deg-daily-no-corr-mean-ft3-bptt-rs0" \
  8 ai2/titan

# Fine-tune: 3-step detached (gradient accumulation, no BPTT), 40 epochs, 8 GPUs on jupiter
run_training \
  "ace-train-config-ft3-detached-daily-fg16-sr0p125-no-corr-mean.yaml" \
  "1deg-daily-no-corr-mean-ft3-detached-rs0" \
  8

# Resume pretrain to 120 epochs (continues wandb gjsqlvsf), 8 GPUs on jupiter
# Job name = original run name so wandb display name is preserved.
run_training \
  "ace-train-config-1-step-pretrain-daily-fg16-sr0p125-no-corr-mean-resume120.yaml" \
  "1deg-daily-no-corr-mean-pretrain-rs0" \
  8

# Fresh 120-epoch pretrain with n_ensemble=3 (vs 2 in gjsqlvsf), 8 GPUs on
# jupiter at LOW beaker priority (the config header also labels CM_PRIORITY=low).
run_training \
  "ace-train-config-1-step-pretrain-daily-fg16-sr0p125-no-corr-mean-nens3-120ep.yaml" \
  "1deg-daily-no-corr-mean-nens3-pretrain-rs0" \
  8 ai2/jupiter low
