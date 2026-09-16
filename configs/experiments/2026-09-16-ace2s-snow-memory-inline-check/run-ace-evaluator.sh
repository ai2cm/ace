#!/bin/bash
# Evaluator runs that exercise the inline anomaly_memory aggregator on the
# control and masked-naive checkpoints, for comparison against the offline
# snow-memory prototype.
#
# Usage:
#   ./run-ace-evaluator.sh              # submit all four
#   ./run-ace-evaluator.sh cm4          # optional substring filter on the job name

set -e

JOB_GROUP="ace2s-snow-memory-inline-check"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
WANDB_USERNAME=${WANDB_USERNAME:-bhenn1983}
REPO_ROOT=$(git rev-parse --show-toplevel)
CLUSTER="${CLUSTER:-ai2/jupiter}"
SELECT="${1:-}"

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

run_evaluator() {
    local arm="$1"
    local ckpt_dataset="$2"
    local job_name="ace2s-snowmem-${arm}-inline-check"
    local config_path="${SCRIPT_PATH}${arm}-evaluator.yaml"

    if [ -n "$SELECT" ] && [[ "$job_name" != *"$SELECT"* ]]; then
        return 0
    fi

    python -m fme.ace.validate_config --config_type evaluator "$config_path"

    gantry run \
        --name $job_name \
        --task-name $job_name \
        --description "ACE2S snow-memory inline check: $arm" \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority normal \
        --min-runtime 8h \
        --cluster "$CLUSTER" \
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
        -- python -I -m fme.ace.evaluator $config_path
}

run_evaluator cm4-control        01KZC1J3R3EW9YVM6HPNSNNNCY
run_evaluator cm4-masked-naive   01KZVBJZ8KHR9E84CEF0NF95ES
run_evaluator era5-control       01KYX6AQTSXD3N23HP128TJYTC
run_evaluator era5-masked-naive  01KZVBA39HPP7ZNZ8FXD2HG9DR
