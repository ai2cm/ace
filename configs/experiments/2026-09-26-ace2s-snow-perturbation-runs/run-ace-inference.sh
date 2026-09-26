#!/bin/bash
# Perturbed-snow sensitivity runs on the fine-tuned CM4 treatment (best-inference epoch 13):
# 30-day inference from custom initial-condition files that differ only in the snow amount,
# eight mid-January members each, logged to the training jobs' W&B group.
#
# Usage:
#   ./run-ace-inference.sh                 # submit all six cases
#   ./run-ace-inference.sh plains          # optional substring filter on the case name

set -e

JOB_GROUP="ace2s-snow-inline-metrics"
IC_DATASET="brianhenn/2026-09-26-ace2s-snow-perturbation-ics-v2"
CKPT_DATASET="01M38TTH492G0WT71YFAEH0V58"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
WANDB_USERNAME=${WANDB_USERNAME:-bhenn1983}
REPO_ROOT=$(git rev-parse --show-toplevel)
CLUSTER="${CLUSTER:-ai2/jupiter}"
SELECT="${1:-}"

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

run_inference() {
    local case="$1"
    local job_name="ace2s-snowpert-${case}"
    local config_path="${SCRIPT_PATH}${case}-inference.yaml"

    if [ -n "$SELECT" ] && [[ "$job_name" != *"$SELECT"* ]]; then
        return 0
    fi

    python -m fme.ace.validate_config --config_type inference "$config_path"

    gantry run \
        --name $job_name \
        --task-name $job_name \
        --description "ACE2S perturbed-snow sensitivity run: $case" \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority normal \
        --min-runtime 1h \
        --cluster "$CLUSTER" \
        --weka climate-default:/climate-default \
        --env WANDB_USERNAME=$WANDB_USERNAME \
        --env WANDB_NAME=$job_name \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP=$JOB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $CKPT_DATASET:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar \
        --dataset $IC_DATASET:/ics \
        --gpus 1 \
        --shared-memory 50GiB \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- python -I -m fme.ace.inference $config_path
}

for case in control null siberia-removed plains-removed plains-plus50 nh-removed; do
    run_inference "$case"
done
