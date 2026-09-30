#!/bin/bash
# Scenario rollouts of the fine-tuned CM4 checkpoints: treatment (masked-naive)
# and control, each run once as the full 140-year 1pctCO2 ramp and once as the
# 40-year piControl holdout tail, four stochastic members per run
# (n_ensemble_per_ic). The model has no CO2 input; it feels the ramp only
# through the prescribed ocean surface.
#
# Usage:
#   ./run-ace-evaluator.sh              # submit all four
#   ./run-ace-evaluator.sh 1pctco2      # optional substring filter on the job name
#   ./run-ace-evaluator.sh control      # the two controls
#
# Dry run (no config edit needed; the evaluator takes --override dotlists):
#   OVERRIDE="n_forward_steps=1000 experiment_dir=gs://vcm-ml-intermediate/2026-09-29-ace2s-snow-scenario-rollouts/dry-run" \
#     ./run-ace-evaluator.sh masked-naive-1pctco2
# Overriding experiment_dir keeps the dry run's small zarr stores out of the
# real output location. Job names get a -dry suffix so W&B stays tidy.

set -e

JOB_GROUP="ace2s-snow-inline-metrics"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
WANDB_USERNAME=${WANDB_USERNAME:-bhenn1983}
REPO_ROOT=$(git rev-parse --show-toplevel)
CLUSTER="${CLUSTER:-ai2/jupiter}"
SELECT="${1:-}"

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

run_evaluator() {
    local arm="$1"
    local scenario="$2"
    local ckpt_dataset="$3"
    local min_runtime="$4"
    local priority="$5"
    local job_name="ace2s-snowmetrics-${arm}-${scenario}-rollout"
    local config_path="${SCRIPT_PATH}${arm}-${scenario}-evaluator.yaml"
    local override_args=()
    if [ -n "${OVERRIDE:-}" ]; then
        job_name="${job_name}-dry"
        override_args=(--override $OVERRIDE)
    fi

    if [ -n "$SELECT" ] && [[ "$job_name" != *"$SELECT"* ]]; then
        return 0
    fi

    python -m fme.ace.validate_config --config_type evaluator "$config_path"

    gantry run \
        --name $job_name \
        --task-name $job_name \
        --description "ACE2S snow scenario rollout: $arm on $scenario" \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority $priority \
        --min-runtime $min_runtime \
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
        -- python -I -m fme.ace.evaluator $config_path "${override_args[@]}"
}

# The piControl jobs (~3 GPU-h) finish inside the 8h min-runtime cap at normal
# priority. The 1pctCO2 jobs (~11 GPU-h plus writes) outlive the protection
# window, so they run urgent to make preemption unlikely; at 1 GPU each this
# should not be too obtrusive to the rest of the team.
run_evaluator cm4-masked-naive picontrol 01M38TTH492G0WT71YFAEH0V58 8h normal
run_evaluator cm4-masked-naive 1pctco2   01M38TTH492G0WT71YFAEH0V58 8h urgent
run_evaluator cm4-control      picontrol 01M33SZ16PFWP822C663RN42Z7 8h normal
run_evaluator cm4-control      1pctco2   01M33SZ16PFWP822C663RN42Z7 8h urgent
