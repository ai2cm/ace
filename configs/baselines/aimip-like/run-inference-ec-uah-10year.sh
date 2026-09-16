#!/bin/bash
# Launch energy-corrector runs with decade-matched unaccounted heating.
# ERA5 budget residual: +4.756 W/m² (1995), +3.994 W/m² (2015).
set -e

JOB_NAME_BASE="ace-4deg-v2-no-residual-no-co2-im4ecamc-10year"
JOB_GROUP="im4ecamc-10year-energy-corrector-uah"
CHECKPOINT_DATASET="01KW0YE54G9NF9YWF000GVPMA8"
CHECKPOINT_FILE="training_checkpoints/best_inference_ckpt.tar"

declare -A UAH
UAH[2015]=3.994
UAH[1995]=4.756

WANDB_USERNAME="${WANDB_USERNAME:-mcgibbon}"
if [[ "$WANDB_USERNAME" != "mcgibbon" ]]; then
    echo "refusing to launch with WANDB_USERNAME=$WANDB_USERNAME" >&2; exit 1
fi

SCRIPT_PATH=$(git rev-parse --show-prefix)
if [[ -z "$SCRIPT_PATH" ]]; then
    echo "run from configs/baselines/aimip-like/" >&2; exit 1
fi
REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT

for YEAR in 2015 1995; do
    if [[ $# -gt 0 && "$YEAR" != "$1" ]]; then continue; fi
    JOB_NAME="${JOB_NAME_BASE}-${YEAR}-raw-ec-uah"
    CONFIG_PATH="${SCRIPT_PATH}/ace-inference-4deg-v2-no-residual-no-co2-10year-${YEAR}-raw-ec.yaml"
    WRAPPER="${SCRIPT_PATH}/run-inference-ec-uah.sh"
    for f in "$CONFIG_PATH" "$WRAPPER"; do
        if [[ ! -f "$f" ]]; then echo "missing $f" >&2; exit 1; fi
    done
    python -m fme.ace.validate_config --config_type inference $CONFIG_PATH

    echo "Launching $JOB_NAME with CONSTANT_UAH=${UAH[$YEAR]}"
    gantry run \
        --name $JOB_NAME \
        --task-name $JOB_NAME \
        --description "Energy-corrector inference (uah=${UAH[$YEAR]}) im4ecamc" \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        --not-preemptible \
        --cluster ai2/jupiter \
        --cluster ai2/titan \
        --env WANDB_USERNAME=$WANDB_USERNAME \
        --env WANDB_NAME=$JOB_NAME \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP=$JOB_GROUP \
        --env CM_PRIORITY=high \
        --env CONSTANT_UAH=${UAH[$YEAR]} \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset $CHECKPOINT_DATASET:$CHECKPOINT_FILE:/ckpt.tar \
        --gpus 1 \
        --shared-memory 50GiB \
        --weka climate-default:/climate-default \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- bash $WRAPPER $CONFIG_PATH
done
