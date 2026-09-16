#!/bin/bash
# Launch energy-corrector 10-year re-run with RH-preserving temperature correction.
# Uses the ec-uah config (unaccounted_heating=3.994) plus preserve_relative_humidity.
set -e
JOB_NAME_BASE="ace-4deg-v2-no-residual-no-co2-im4ecamc-10year"
JOB_GROUP="im4ecamc-10year-energy-corrector-uah-rh"
CHECKPOINT_DATASET="01KW0YE54G9NF9YWF000GVPMA8"
CHECKPOINT_FILE="training_checkpoints/best_inference_ckpt.tar"
WANDB_USERNAME="${WANDB_USERNAME:-mcgibbon}"
if [[ "$WANDB_USERNAME" != "mcgibbon" ]]; then echo "bad WANDB_USERNAME" >&2; exit 1; fi
SCRIPT_PATH=$(git rev-parse --show-prefix)
[[ -z "$SCRIPT_PATH" ]] && { echo "run from configs/baselines/aimip-like/" >&2; exit 1; }
cd "$(git rev-parse --show-toplevel)"
YEAR=2015
JOB_NAME="${JOB_NAME_BASE}-${YEAR}-raw-ec-uah-rh"
CONFIG="${SCRIPT_PATH}/ace-inference-4deg-v2-no-residual-no-co2-10year-${YEAR}-raw-ec-uah-rh.yaml"
python -m fme.ace.validate_config --config_type inference "$CONFIG"
echo "Launching $JOB_NAME"
gantry run --name "$JOB_NAME" --task-name "$JOB_NAME" \
    --description "Energy-corrector inference (uah=ERA5 + RH-preserving) im4ecamc" \
    --beaker-image "$(cat latest_deps_only_image.txt)" \
    --workspace ai2/ace --priority high --not-preemptible \
    --cluster ai2/jupiter --cluster ai2/titan \
    --env WANDB_USERNAME=$WANDB_USERNAME --env WANDB_NAME="$JOB_NAME" \
    --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=$JOB_GROUP \
    --env CM_PRIORITY=high --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset "$CHECKPOINT_DATASET:$CHECKPOINT_FILE:/ckpt.tar" \
    --gpus 1 --shared-memory 50GiB --weka climate-default:/climate-default \
    --budget ai2/atec-climate --system-python --install "pip install --no-deps ." \
    -- python -I -m fme.ace.inference "$CONFIG"
