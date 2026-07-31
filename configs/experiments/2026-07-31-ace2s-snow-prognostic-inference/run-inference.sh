#!/bin/bash
# Standard CM4-piControl 40-year evaluation of the ACE2S snow-prognostic model
# (see configs/experiments/2026-07-14-ace2s-snow-prognostic-training/).
#
# Single-IC, 58,300-step (~39.9 yr at 6h) offline rollout over the holdout period,
# starting at the train/validation/holdout boundary 0311-01-01T06:00:00.
#
# Usage:
#   ./run-inference.sh

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)   # this dir, relative to the repo root
WANDB_USERNAME=${WANDB_USERNAME:-bhenn1983}  # W&B handle (differs from the Beaker username)
REPO_ROOT=$(git rev-parse --show-toplevel)
JOB_GROUP="ace2s-snow-prognostic"
DATA_TAG="1deg-6h"                           # data resolution + timestep of the evaluated model
CONFIG_PATH="${SCRIPT_PATH}cm4.yaml"

# Beaker result dataset of the CM4 snow-prognostic 1-step pretrain
# (job ace2s-snowprog-cm4-1deg-6h-pretrain, beaker experiment 01KXPE9HE4BC5M7D2DGFRME5JF).
CKPT_DATASET=01KXPJ21JHMQ7YHJ0GTZZ65APC
JOB_NAME="ace2s-snowprog-cm4-${DATA_TAG}-evaluator"

cd "$REPO_ROOT"  # so the config path resolves regardless of where this is run from

python -m fme.ace.validate_config --config_type evaluator "$CONFIG_PATH"

gantry run \
  --name "$JOB_NAME" \
  --task-name "$JOB_NAME" \
  --description "ACE2S snow-prognostic CM4 40-year evaluation" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace \
  --priority normal \
  --not-preemptible \
  --cluster ai2/titan \
  --weka climate-default:/climate-default \
  --dataset "$CKPT_DATASET:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar" \
  --env WANDB_USERNAME="$WANDB_USERNAME" \
  --env WANDB_NAME="$JOB_NAME" \
  --env WANDB_JOB_TYPE=inference \
  --env WANDB_RUN_GROUP="$JOB_GROUP" \
  --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset-secret google-credentials:/tmp/google_application_credentials.json \
  --gpus 1 \
  --shared-memory 50GiB \
  --budget ai2/atec-climate \
  --system-python \
  --install "pip install --no-deps ." \
  -- python -I -m fme.ace.evaluator "$CONFIG_PATH"
