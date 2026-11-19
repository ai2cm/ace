#!/bin/bash
#
# Launches the constant-CO2 control rollout of the CO2-input baseline.
# Run FROM this configs directory. Inference has no checkpointing, so the job
# is not preemptible.

set -euo pipefail

WANDB_IDENTITY="mcgibbon"
WANDB_USERNAME=${WANDB_USERNAME:-$WANDB_IDENTITY}
if [[ "$WANDB_USERNAME" != "$WANDB_IDENTITY" ]]; then
  echo "ERROR: WANDB_USERNAME='$WANDB_USERNAME' but runs must attribute to '$WANDB_IDENTITY'." >&2
  exit 1
fi
SCRIPT_PATH=$(git rev-parse --show-prefix)
REPO_ROOT=$(git rev-parse --show-toplevel)
if [[ -z "$SCRIPT_PATH" ]]; then
  echo "ERROR: invoke run-evaluator.sh FROM its own configs directory." >&2
  exit 1
fi

JOB_NAME="muon-4deg-pretrain-rs1-ep120-81year-constant-co2"
CONFIG_PATH="$SCRIPT_PATH/evaluator-baseline-constant-co2.yaml"
# result dataset of the final job of beaker experiment
# 01M70VEE6375A7G3X0SSV97HDG (muon-4deg-pretrain-rs1, wandb b6v0uw1j)
CHECKPOINT_DATASET=01M7536HDXJ6MDGVXAXZKNCJ2M

cd "$REPO_ROOT"
python -m fme.ace.validate_config --config_type evaluator "$CONFIG_PATH"

gantry run \
  --name "$JOB_NAME" \
  --description 'Constant-CO2 81-year rollout of the CO2-input 4deg baseline' \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace \
  --priority "${PRIORITY:-high}" \
  --not-preemptible \
  --cluster ai2/jupiter \
  --cluster ai2/titan \
  --env WANDB_USERNAME="$WANDB_USERNAME" \
  --env WANDB_NAME="$JOB_NAME" \
  --env WANDB_JOB_TYPE=inference \
  --env WANDB_RUN_GROUP= \
  --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset-secret google-credentials:/tmp/google_application_credentials.json \
  --dataset "$CHECKPOINT_DATASET:training_checkpoints/ema_ckpt_0120.tar:/ckpt.tar" \
  --gpus 1 \
  --shared-memory 50GiB \
  --weka climate-default:/climate-default \
  --budget ai2/atec-climate \
  --allow-dirty \
  --system-python \
  --install "pip install --no-deps ." \
  -- python -I -m fme.ace.evaluator "$CONFIG_PATH"
