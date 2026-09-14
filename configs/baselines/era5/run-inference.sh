#!/bin/bash
#
# Inference launcher for daily 1-degree ERA5 candidates.
# Runs standalone evaluator jobs for three models:
#   - paper-recipe BPTT fine-tune (c8hp09jm)
#   - no-corr+mean BPTT fine-tune (fo3yeiew)
#   - no-corr+mean detached fine-tune (z9o2mjt7)
#
# Each model gets 7 inference runs: 10yr×3ICs, 81yr×3ICs, weather-2020.
#
# Usage (run FROM this directory):
#   ./run-inference.sh                    # launch all 21 runs
#   ./run-inference.sh paper-recipe       # only paper-recipe model
#   ./run-inference.sh nocorr-bptt 81yr   # only nocorr-bptt 81yr runs
#   ./run-inference.sh 10yr               # only 10yr runs for all models

set -euo pipefail

# === GUARDRAILS (copy verbatim from the reference; do not hand-edit) =========
WANDB_IDENTITY="mcgibbon"

SCRIPT_PATH=$(git rev-parse --show-prefix)
REPO_ROOT=$(git rev-parse --show-toplevel)

WANDB_USERNAME=${WANDB_USERNAME:-$WANDB_IDENTITY}
if [[ "$WANDB_USERNAME" != "$WANDB_IDENTITY" ]]; then
  echo "ERROR: WANDB_USERNAME='$WANDB_USERNAME' but runs must attribute to '$WANDB_IDENTITY'." >&2
  exit 1
fi

if [[ -z "$SCRIPT_PATH" ]]; then
  echo "ERROR: SCRIPT_PATH is empty. Run from the configs directory." >&2
  exit 1
fi

LAUNCH_FILTERS=("$@")
should_run() {
  [[ ${#LAUNCH_FILTERS[@]} -eq 0 ]] && return 0
  local f
  for f in "${LAUNCH_FILTERS[@]}"; do
    [[ "$1" == *"$f"* ]] && return 0
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

# Eval configs beaker dataset (daily 1deg, two-store merge)
EVAL_CONFIGS_DATASET="01M2GQ0F2Y0129X3M030FMY3S5"

# Checkpoint result datasets
declare -A CKPT_DATASETS=(
  [paper-recipe]="01M1FAC2CXBCB12S3XJC2WP2YY"
  [nocorr-bptt]="01M23HMDHMBP1GEAM7N4575N83"
  [nocorr-detached]="01M28YHEQAC9H2TZ6EYG3Z6C4V"
)

# wandb run groups
declare -A WANDB_GROUPS=(
  [paper-recipe]="paper-recipe-c8hp09jm-inference"
  [nocorr-bptt]="nocorr-mean-bptt-fo3yeiew-inference"
  [nocorr-detached]="nocorr-mean-detached-z9o2mjt7-inference"
)

run_inference() {
  local model="$1"
  local eval_name="$2"  # e.g. 10yr-IC0, 81yr-IC1, weather-2020
  local job_name="${model}-${eval_name}"

  should_run "$job_name" || { echo "skip (filter): $job_name"; return 0; }

  local ckpt_dataset="${CKPT_DATASETS[$model]}"
  local wandb_group="${WANDB_GROUPS[$model]}"
  local eval_config="evaluator-${eval_name}.yaml"

  echo "launching: $job_name  (ckpt=$ckpt_dataset  config=$eval_config)"

  gantry run \
    --name "$job_name" \
    --description "Inference: $model $eval_name" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace \
    --priority high \
    --min-runtime 8h \
    --not-preemptible \
    --cluster ai2/jupiter \
    --cluster ai2/titan \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP="$wandb_group" \
    --env CM_PRIORITY=high \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "$ckpt_dataset":training_checkpoints/best_ckpt.tar:/ckpt.tar \
    --dataset "$EVAL_CONFIGS_DATASET":/eval-configs \
    --gpus 1 \
    --shared-memory 50GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --allow-dirty \
    --system-python \
    --install "pip install --no-deps ." \
    -- python -I -m fme.ace.evaluator "/eval-configs/$eval_config"
}

# Paper-recipe candidate (c8hp09jm)
run_inference paper-recipe 10yr-IC0
run_inference paper-recipe 10yr-IC1
run_inference paper-recipe 10yr-IC2
run_inference paper-recipe 81yr-IC0
run_inference paper-recipe 81yr-IC1
run_inference paper-recipe 81yr-IC2
run_inference paper-recipe weather-2020

# No-corr+mean BPTT fine-tune (fo3yeiew)
run_inference nocorr-bptt 10yr-IC0
run_inference nocorr-bptt 10yr-IC1
run_inference nocorr-bptt 10yr-IC2
run_inference nocorr-bptt 81yr-IC0
run_inference nocorr-bptt 81yr-IC1
run_inference nocorr-bptt 81yr-IC2
run_inference nocorr-bptt weather-2020

# No-corr+mean detached fine-tune (z9o2mjt7)
run_inference nocorr-detached 10yr-IC0
run_inference nocorr-detached 10yr-IC1
run_inference nocorr-detached 10yr-IC2
run_inference nocorr-detached 81yr-IC0
run_inference nocorr-detached 81yr-IC1
run_inference nocorr-detached 81yr-IC2
run_inference nocorr-detached weather-2020
