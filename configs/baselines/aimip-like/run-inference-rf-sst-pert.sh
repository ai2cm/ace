#!/bin/bash
#
# Perturbed-SST (+0/+2/+4 K) inference for the receptive-field cohort's local
# checkpoints, mirroring the ace2s-4deg-daily-era5-only-rs0-no-norm-baseline-p*k
# runs (which cover the matched global checkpoint, train-4deg-daily-era5-only-
# no-co2-rs0 / wandb injiirnf). Same p0k/p2k/p4k configs; only the mounted
# checkpoint and job name vary.
#
# Usage (run FROM this configs directory):
#   ./run-inference-rf-sst-pert.sh              # all 9 jobs
#   ./run-inference-rf-sst-pert.sh ankur        # only jobs whose name matches

set -euo pipefail

# === GUARDRAILS (from research run-train.reference.sh) ======================
WANDB_IDENTITY="mcgibbon"

SCRIPT_PATH=$(git rev-parse --show-prefix)
REPO_ROOT=$(git rev-parse --show-toplevel)
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')

WANDB_USERNAME=${WANDB_USERNAME:-$WANDB_IDENTITY}
if [[ "$WANDB_USERNAME" != "$WANDB_IDENTITY" ]]; then
  echo "ERROR: WANDB_USERNAME='$WANDB_USERNAME' but runs must attribute to '$WANDB_IDENTITY'." >&2
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

JOB_GROUP="receptive-field-sst-perts"

cd "$REPO_ROOT"

for pert in p0k p2k p4k; do
  python -m fme.ace.validate_config --config_type inference "$SCRIPT_PATH/ace-inference-era5-$pert.yaml"
done

launch_job () {
  local JOB_NAME=$1
  local CONFIG_PATH=$2
  local CKPT_DATASET=$3

  should_run "$JOB_NAME" || { echo "skip (filter): $JOB_NAME"; return 0; }
  echo "launching: $JOB_NAME"

  gantry run \
    --name "$JOB_NAME" \
    --task-name "$JOB_NAME" \
    --description 'Perturbed-SST inference, receptive-field cohort' \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace \
    --priority high \
    --not-preemptible \
    --cluster ai2/jupiter \
    --cluster ai2/titan \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$JOB_NAME" \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP="$JOB_GROUP" \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "$CKPT_DATASET":training_checkpoints/best_inference_ckpt.tar:/ckpt.tar \
    --gpus 1 \
    --shared-memory 50GiB \
    --allow-dirty \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- python -I -m fme.ace.inference "$CONFIG_PATH"
}

# checkpoint datasets: best_inference_ckpt.tar of the finished training runs
CKPT_LOCAL_HEAD="01KS61ERSWPD3B4HAN98VSVJ06"   # local 1x1 head    (wandb 8vdp7f8k)
CKPT_ANKUR_HEAD="01KS61FDG86SRTXH1J7CYGFWGJ"   # DISCO 3x3 head    (wandb cpicisue)
CKPT_FULLY_LOCAL="01KS61E5224XRR8GM086JG9YXX"  # fully local       (wandb 4ouash9m)

for pert in p0k p2k p4k; do
  launch_job "ace2s-4deg-daily-era5-only-local-mlp-diagnostics-$pert-v2"       "$SCRIPT_PATH/ace-inference-era5-$pert.yaml" "$CKPT_LOCAL_HEAD"
  launch_job "ace2s-4deg-daily-era5-only-ankur-local-mlp-diagnostics-$pert-v2" "$SCRIPT_PATH/ace-inference-era5-$pert.yaml" "$CKPT_ANKUR_HEAD"
  launch_job "ace2s-4deg-daily-era5-only-local-mlp-$pert-v2"                   "$SCRIPT_PATH/ace-inference-era5-$pert.yaml" "$CKPT_FULLY_LOCAL"
done
