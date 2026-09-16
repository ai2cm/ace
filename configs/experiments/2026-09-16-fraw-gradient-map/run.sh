#!/bin/bash
# dL/dF_raw through the force-positive clamp and the frozen clip at a
# pre-corrector fine-tuning checkpoint. One GPU, a few val_piC batches; writes
# fraw_gradient_map.nc and facts.json to /results.
#
#   ./run.sh fprec        wandb kj3ap00o, precorrector on PRATEsfc, adv, total_frozen_precipitation_rate
#   ./run.sh prate-adv    wandb qdpcwkgr, precorrector on PRATEsfc, adv
#
# Both at training_checkpoints/best_inference_ckpt.tar.

set -e

VARIANT="${1:?variant: fprec | prate-adv}"
case "$VARIANT" in
    fprec)     TRAINING_RESULTS_DATASET="01M2K5X75KYPVXCVRJNRFAHYPZ" ;;  # kj3ap00o
    prate-adv) TRAINING_RESULTS_DATASET="01M2KABFYBNCT0EDJ9HZP07F7Q" ;;  # qdpcwkgr
    *) echo "unknown variant $VARIANT"; exit 1 ;;
esac
CKPT_TYPE="best_inference_ckpt"
JOB_NAME="fraw-gradient-map-${VARIANT}-${CKPT_TYPE//_/-}"
JOB_GROUP="2026-09-16-fraw-gradient-map"
ATMOS_STATS_DATASET="01KXNT0RA6VX2YTZ8WJ936Q5RS"      # as mounted by the training runs

REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_DIR=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')

cd "$REPO_ROOT"

gantry run \
    --name "$JOB_NAME" \
    --task-name "$JOB_NAME" \
    --description "dL/dF_raw through the frozen-precip clamp and clip, ft_precorr $VARIANT $CKPT_TYPE, val_piC" \
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
    -- python -I "$SCRIPT_DIR/fraw_gradient_map.py" "$SCRIPT_DIR/config-$VARIANT.yaml"
