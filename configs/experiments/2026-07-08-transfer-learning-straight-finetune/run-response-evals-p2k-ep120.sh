#!/bin/bash
#
# +2K SST fill-in for the end-of-training (epoch 120) checkpoints of the two
# NON-exclude-global-MLP arms: the corrected no-residual modern fine-tune and
# the original-style reproduction fine-tune. The epoch sweep
# (run-response-evals-sweep.sh) ran p0k + p4k only, so those arms have no
# +2K point; the exclude-global-MLP arms and the donor do. Without these two
# jobs the response-vs-amplitude (linearity) comparison cannot put the
# exclude-global-MLP arms next to the config they were built on at a matched
# checkpoint.
#
# 2 jobs: {no-residual, reproduction} x p2k x ep120.
#
# research: tasks/2026-07-08-finetune-era5-excluding-sfno-global-mlp.md
#
# Usage (run FROM this configs directory):
#   ./run-response-evals-p2k-ep120.sh                    # both jobs
#   ./run-response-evals-p2k-ep120.sh no-residual        # substring filter

set -euo pipefail

# === GUARDRAILS (from research run-train.reference.sh) ======================
WANDB_IDENTITY="mcgibbon"
SCRIPT_PATH=$(git rev-parse --show-prefix)
REPO_ROOT=$(git rev-parse --show-toplevel)
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-$WANDB_IDENTITY}
if [[ "$WANDB_USERNAME" != "$WANDB_IDENTITY" ]]; then
  echo "ERROR: WANDB_USERNAME='$WANDB_USERNAME' but runs must attribute to '$WANDB_IDENTITY'." >&2
  echo "       (BEAKER_USERNAME='$BEAKER_USERNAME' would misattribute to the wandb service account.)" >&2
  exit 1
fi
if [[ -z "$SCRIPT_PATH" ]]; then
  echo "ERROR: run this script from its own configs directory, not the repo root." >&2
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
# === END GUARDRAILS ==========================================================

JOB_GROUP="tl-straight-finetune-response-evals-sweep"
EMA=120
declare -A CKPT=(
  [reproduction]="01KXPP4WPXNVWRFZCJRG4EJKEA"
  [no-residual]="01KY2T41ARVW0EZ7SS267THKB9"
)

cd "$REPO_ROOT"

python -m fme.ace.validate_config --config_type inference "$SCRIPT_PATH/ace-inference-era5-p2k.yaml"

gantry_common=(
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")"
  --workspace ai2/ace
  --priority high
  --not-preemptible
  --cluster ai2/jupiter
  --cluster ai2/titan
  --env WANDB_USERNAME="$WANDB_USERNAME"
  --env WANDB_JOB_TYPE=inference
  --env WANDB_RUN_GROUP="$JOB_GROUP"
  --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa
  --dataset-secret google-credentials:/tmp/google_application_credentials.json
  --gpus 1
  --shared-memory 50GiB
  --allow-dirty
  --weka climate-default:/climate-default
  --budget ai2/atec-climate
  --system-python
  --install "pip install --no-deps ."
)

launch () {   # launch <job_name> <ckpt_dataset>
  local JOB_NAME=$1 CKPT_DATASET=$2
  should_run "$JOB_NAME" || { echo "skip (filter): $JOB_NAME"; return 0; }
  echo "launching: $JOB_NAME"
  gantry run \
    --name "$JOB_NAME" \
    --task-name "$JOB_NAME" \
    --description '+2K fill-in at ep120 for the non-exclude-global-MLP arms' \
    --env WANDB_NAME="$JOB_NAME" \
    "${gantry_common[@]}" \
    --dataset "$CKPT_DATASET":"training_checkpoints/ema_ckpt_0$EMA.tar":/ckpt.tar \
    -- python -I -m fme.ace.inference "$SCRIPT_PATH/ace-inference-era5-p2k.yaml"
}

for arm in no-residual reproduction; do
  launch "sweep-era5-finetune-$arm-p2k-ep$EMA" "${CKPT[$arm]}"
done
