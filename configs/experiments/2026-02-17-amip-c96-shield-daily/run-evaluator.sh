#!/bin/bash

set -e

JOB_NAME="ace-evaluator"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

run_evaluator() {
    local config_filename="$1"
    local dataset_id="$2"
    local job_name="$3"
    local ckpt_filename="${4:-best_inference_ckpt.tar}"
    local CONFIG_PATH="$SCRIPT_PATH/$config_filename"
    python -m fme.ace.validate_config --config_type evaluator $CONFIG_PATH

    cd $REPO_ROOT && gantry run \
        --name $job_name \
        --task-name $job_name \
        --description 'Run ace evaluator' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/climate-titan \
        --priority urgent \
        --not-preemptible \
        --cluster ai2/titan \
        --env WANDB_USERNAME=$WANDB_USERNAME \
        --env WANDB_NAME=$job_name \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP= \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $dataset_id:training_checkpoints/$ckpt_filename:/ckpt.tar \
        --gpus 1 \
        --shared-memory 50GiB \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --system-python \
        --allow-dirty \
        --install "pip install --no-deps ." \
        -- python -I -m fme.ace.evaluator $CONFIG_PATH
}

# run_evaluator "evaluator-6h.yaml" "01KJKM724X54EMA1JFEE5JWX9V" "amip-c96-shield-evaluator-6h"
# run_evaluator "evaluator-daily.yaml" "01KJCVYVEB52QATDCZ3ZA74TDT" "amip-c96-shield-evaluator-daily"
run_evaluator "evaluator-6h.yaml" "01KJKM724X54EMA1JFEE5JWX9V" "amip-c96-shield-evaluator-6h-best-ckpt" "best_ckpt.tar"
run_evaluator "evaluator-daily.yaml" "01KJCVYVEB52QATDCZ3ZA74TDT" "amip-c96-shield-evaluator-daily-best-ckpt" "best_ckpt.tar"
