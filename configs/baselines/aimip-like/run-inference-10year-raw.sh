#!/bin/bash
# Launch the standalone raw-output 10-year re-runs of the no-CO2, no-residual
# 4deg v2 model (wandb im4ecamc): the 2015-2024 out-of-sample decade and its
# 1995-2004 in-sample twin. Run from configs/baselines/aimip-like/.
set -e

JOB_NAME_BASE="ace-4deg-v2-no-residual-no-co2-im4ecamc-10year"
JOB_GROUP="im4ecamc-10year-raw-output"
# Result dataset of the im4ecamc training job (beaker experiment 01KW01ARRGSMYG4K8YPPGR33C8).
CHECKPOINT_DATASET="01KW0YE54G9NF9YWF000GVPMA8"
CHECKPOINT_FILE="training_checkpoints/best_inference_ckpt.tar"

# wandb attribution: the beaker job env does not carry WANDB_USERNAME, and the
# default (the beaker account, jeremym) misattributes the run.
WANDB_USERNAME="${WANDB_USERNAME:-mcgibbon}"
if [[ "$WANDB_USERNAME" != "mcgibbon" ]]; then
    echo "refusing to launch with WANDB_USERNAME=$WANDB_USERNAME (expected mcgibbon)" >&2
    exit 1
fi

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
if [[ -z "$SCRIPT_PATH" ]]; then
    echo "run this script from its own directory (configs/baselines/aimip-like/)" >&2
    exit 1
fi
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config paths are valid no matter where we are running this script

launch_job () {
    JOB_NAME=$1
    CONFIG_PATH=$2
    if [[ ! -f "$CONFIG_PATH" ]]; then
        echo "missing config $CONFIG_PATH" >&2
        exit 1
    fi
    python -m fme.ace.validate_config --config_type inference $CONFIG_PATH

    # Inference has no checkpointing, so a preempted job loses everything:
    # --not-preemptible.
    gantry run \
        --name $JOB_NAME \
        --task-name $JOB_NAME \
        --description 'Raw-output 10-year ACE inference re-run (im4ecamc)' \
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
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset $CHECKPOINT_DATASET:$CHECKPOINT_FILE:/ckpt.tar \
        --gpus 1 \
        --shared-memory 50GiB \
        --weka climate-default:/climate-default \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- python -I -m fme.ace.inference $CONFIG_PATH
}

for YEAR in 2015 1995; do
    if [[ $# -gt 0 && "$YEAR" != "$1" ]]; then continue; fi
    JOB_NAME="${JOB_NAME_BASE}-${YEAR}-raw"
    echo "Launching job: $JOB_NAME"
    launch_job "$JOB_NAME" "$SCRIPT_PATH/ace-inference-4deg-v2-no-residual-no-co2-10year-${YEAR}-raw.yaml"
done
