#!/bin/bash
# Standard CM4-piControl 40-year evaluation, run once per training stage of the
# two-stage ACE2S recipe. Both stages are evaluated because they are the two
# candidate baselines for the snow-prognostic treatment
# (configs/experiments/2026-07-31-ace2s-snow-prognostic-inference/): the pretrain
# is stage-matched to that treatment's checkpoint, the finetune is the full recipe.
#
# Usage:
#   ./run-ace-evaluator.sh                # submit both arms
#   ./run-ace-evaluator.sh pretrain       # optional substring filter on the job name

set -e

JOB_GROUP="ace2s-cm4-picontrol"
CONFIG_FILENAME="ace-evaluator-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH="${SCRIPT_PATH}${CONFIG_FILENAME}"
WANDB_USERNAME=${WANDB_USERNAME:-bhenn1983}
REPO_ROOT=$(git rev-parse --show-toplevel)
SELECT="${1:-}"  # optional substring filter on job name

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

python -m fme.ace.validate_config --config_type evaluator $CONFIG_PATH

run_evaluator() {
    local ckpt_dataset="$1"
    local job_name="$2"

    if [ -n "$SELECT" ] && [[ "$job_name" != *"$SELECT"* ]]; then
        return 0
    fi

    gantry run \
        --name $job_name \
        --task-name $job_name \
        --description 'Run ACE2S CM4 piControl atmosphere evaluator' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority normal \
        --not-preemptible \
        --cluster ai2/ceres \
        --cluster ai2/jupiter \
        --cluster ai2/saturn \
        --weka climate-default:/climate-default \
        --env WANDB_USERNAME=$WANDB_USERNAME \
        --env WANDB_NAME=$job_name \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP=$JOB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $ckpt_dataset:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar \
        --gpus 1 \
        --shared-memory 50GiB \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- python -I -m fme.ace.evaluator $CONFIG_PATH
}

# Stage-1 1-step pretrain (run ace2s-cm4-picontrol-1-step-pretrain-rs0, stopped at epoch 11).
run_evaluator "01KTPTS6C23P8SWB9RBFWB09BE" "ace2s-cm4-picontrol-evaluator-pretrain-rs0"
# Stage-2 multi-step finetune, warm-started from the pretrain above
# (run ace2s-cm4-picontrol-multi-step-finetuning-rs0).
run_evaluator "01KTYXNSJX90Y5E2CQ6SV8K37D" "ace2s-cm4-picontrol-evaluator-finetune-rs0"
