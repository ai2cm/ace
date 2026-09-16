#!/bin/bash
# dT and d(dT)/d(alpha) through the frozen clip at an 84vk67xi checkpoint. One GPU,
# a few val_piC batches; writes dtdalpha_map.nc and facts.json to /results.
#
#   ./run.sh [best_ckpt|best_inference_ckpt|ckpt]     default best_inference_ckpt
#
# best_* hold the EMA weights; ckpt holds the optimized weights at the last epoch.

set -e

CKPT_TYPE="${1:-best_inference_ckpt}"
JOB_NAME="dtdalpha-map-84vk67xi-${CKPT_TYPE//_/-}"
JOB_GROUP="2026-09-16-dtdalpha-map"
TRAINING_RESULTS_DATASET="01M1CV3XCNJQ225GT89ES8SM18"  # beaker results dataset of wandb run 84vk67xi
ATMOS_STATS_DATASET="01KXNT0RA6VX2YTZ8WJ936Q5RS"      # as mounted by the training run

REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_DIR=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')

cd "$REPO_ROOT"

gantry run \
    --name "$JOB_NAME" \
    --task-name "$JOB_NAME" \
    --description "dT and d(dT)/d(alpha) through the frozen clip, 84vk67xi $CKPT_TYPE, val_piC" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace \
    --priority high \
    --cluster ai2/jupiter \
    --weka climate-default:/climate-default \
    --env WANDB_USERNAME="$BEAKER_USERNAME" \
    --env WANDB_NAME="$JOB_NAME" \
    --env WANDB_JOB_TYPE=analysis \
    --env WANDB_RUN_GROUP="$JOB_GROUP" \
    --env WANDB_MODE=disabled \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "$ATMOS_STATS_DATASET:coupled_atmosphere:/atmos_stats" \
    --dataset "$TRAINING_RESULTS_DATASET:training_checkpoints/$CKPT_TYPE.tar:/ckpt.tar" \
    --gpus 1 \
    --shared-memory 20GiB \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- python -I "$SCRIPT_DIR/dtdalpha_map.py" "$SCRIPT_DIR/config.yaml"
