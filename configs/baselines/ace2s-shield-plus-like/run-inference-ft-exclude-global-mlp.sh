#!/bin/bash
#
# Post-hoc +0/+2/+4 K constant-SST-perturbation inference for the SHiELD+ -> ERA5
# fine-tune, family-1 variant C (exclude the SFNO global MLP). The +4K response
# axis of the transfer-learning verdict goal; the future-scenario axis is the
# inline long_46year / long_46year_constant_co2 cases in the train configs.
#
# research: tasks/2026-07-08-finetune-era5-excluding-sfno-global-mlp.md
#
# Run this AFTER the two fine-tune training runs finish. Fill CKPT_LR1E4 /
# CKPT_LR1E5 below with each run's beaker result-dataset id (from
# `rsrch sync` / the experiment record), then:
#   ./run-inference-ft-exclude-global-mlp.sh            # all 6 jobs
#   ./run-inference-ft-exclude-global-mlp.sh lr1e-5     # only the lr1e-5 arm
#
# CKPT_FILE selects which in-dataset checkpoint to evaluate (default
# best_inference_ckpt.tar); NAME_SUFFIX disambiguates the beaker/wandb run
# names and group when evaluating a second checkpoint. The end-of-training
# eval (the one that shows whether the full fine-tune forgot the extrapolant
# response) is:
#   CKPT_FILE=training_checkpoints/ema_ckpt_0120.tar NAME_SUFFIX=-ep120 \
#     ./run-inference-ft-exclude-global-mlp.sh
# (best_inference_ckpt landed at epoch 4 for lr1e-4 / epoch 100 for lr1e-5, so
# it does not answer the end-of-training question for the aggressive arm.)

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

CKPT_FILE="${CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
NAME_SUFFIX="${NAME_SUFFIX:-}"
JOB_GROUP="finetune-era5-exclude-global-mlp-sst-perts${NAME_SUFFIX}"

# Fine-tune result datasets (best_inference_ckpt.tar). FILL AFTER TRAINING.
CKPT_LR1E4="01KXPK1RXTK98KBGES4AY8K9KY"   # ft-...-lr1e-4-rs0 result dataset (wandb h05vdqxj, epoch 120)
CKPT_LR1E5="01KXSA9V4A1SQ1M5R5MBDVJZ8B"   # ft-...-lr1e-5-rs0 result dataset (wandb fizm7uqz, epoch 120)

cd "$REPO_ROOT"

for pert in p0k p2k p4k; do
  python -m fme.ace.validate_config --config_type inference "$SCRIPT_PATH/ace-inference-era5-$pert.yaml"
done

launch_job () {
  local JOB_NAME=$1
  local CONFIG_PATH=$2
  local CKPT_DATASET=$3

  should_run "$JOB_NAME" || { echo "skip (filter): $JOB_NAME"; return 0; }
  if [[ "$CKPT_DATASET" == "TBD" ]]; then
    echo "ERROR: checkpoint dataset for $JOB_NAME is still TBD — fill CKPT_LR1E4/CKPT_LR1E5." >&2
    exit 1
  fi
  echo "launching: $JOB_NAME"

  gantry run \
    --name "$JOB_NAME" \
    --task-name "$JOB_NAME" \
    --description 'Perturbed-SST inference, SHiELD+->ERA5 FT exclude global MLP' \
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
    --dataset "$CKPT_DATASET":"$CKPT_FILE":/ckpt.tar \
    --gpus 1 \
    --shared-memory 50GiB \
    --allow-dirty \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- python -I -m fme.ace.inference "$CONFIG_PATH"
}

for pert in p0k p2k p4k; do
  launch_job "ft-4deg-daily-era5-exclude-global-mlp-lr1e-4-$pert$NAME_SUFFIX" "$SCRIPT_PATH/ace-inference-era5-$pert.yaml" "$CKPT_LR1E4"
  launch_job "ft-4deg-daily-era5-exclude-global-mlp-lr1e-5-$pert$NAME_SUFFIX" "$SCRIPT_PATH/ace-inference-era5-$pert.yaml" "$CKPT_LR1E5"
done
