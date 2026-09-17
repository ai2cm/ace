#!/bin/bash
# Launch energy-corrector 10-year re-runs (stepper_override.corrector, uah=0).
set -e
JOB_NAME_BASE="ace-4deg-v2-no-co2-2ki61qpw-10year"
JOB_GROUP="2ki61qpw-10year-energy-corrector"
CHECKPOINT_DATASET="01KVXJTY7KVT7X1SQZE3YW5NB1"
CHECKPOINT_FILE="training_checkpoints/best_inference_ckpt.tar"
WANDB_USERNAME="${WANDB_USERNAME:-mcgibbon}"
if [[ "$WANDB_USERNAME" != "mcgibbon" ]]; then echo "bad WANDB_USERNAME" >&2; exit 1; fi
SCRIPT_PATH=$(git rev-parse --show-prefix)
[[ -z "$SCRIPT_PATH" ]] && { echo "run from configs/baselines/aimip-like/" >&2; exit 1; }
cd "$(git rev-parse --show-toplevel)"
for YEAR in 2015 1995; do
    [[ $# -gt 0 && "$YEAR" != "$1" ]] && continue
    JOB_NAME="${JOB_NAME_BASE}-${YEAR}-raw-ec"
    CONFIG="${SCRIPT_PATH}/ace-inference-4deg-v2-no-co2-10year-${YEAR}-raw-ec.yaml"
    python -m fme.ace.validate_config --config_type inference "$CONFIG"
    echo "Launching $JOB_NAME"
    gantry run --name "$JOB_NAME" --task-name "$JOB_NAME" \
        --description "Energy-corrector inference (uah=0) 2ki61qpw" \
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
done
