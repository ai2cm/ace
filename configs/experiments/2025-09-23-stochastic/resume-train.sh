#!/bin/bash

set -e

# Configuration variables
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
# Since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=8

# Override arguments (add any config overrides here)
OVERRIDE_ARGS=""

# Change to the repo root so paths are valid no matter where we run the script from
cd "$REPO_ROOT"

resume_training() {
    local config_filename="$1"
    local job_name="$2"
    local existing_results_dataset="$3"
    local CONFIG_PATH="$SCRIPT_PATH/$config_filename"

    python -m fme.ace.validate_config --config_type train "$CONFIG_PATH"

    # Extract additional args from config header
    local extra_args=()
    while IFS= read -r line; do
        [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
    done < "$CONFIG_PATH"

    gantry run \
        --name "resume-${job_name}" \
        --description "Resume training" \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        --preemptible \
        --cluster ai2/ceres \
        --cluster ai2/jupiter \
        --weka climate-default:/climate-default \
        --env WANDB_USERNAME="$WANDB_USERNAME" \
        --env WANDB_NAME="$job_name" \
        --env WANDB_JOB_TYPE=training \
        --env WANDB_RUN_GROUP= \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset "$existing_results_dataset:/existing-results" \
        --gpus "$N_GPUS" \
        --shared-memory 400GiB \
        --budget ai2/climate \
        --system-python \
        --install "pip install --no-deps ." \
        "${extra_args[@]}" \
        -- torchrun --nproc_per_node "$N_GPUS" -m fme.ace.train /existing-results/config.yaml \
        --override resume_results.existing_dir=/existing-results $OVERRIDE_ARGS
}

base_name="stochastic"

# resume_training "train-era5-n384-e9c1-gauss-1step.yaml" "$base_name-era5-n384-e9c1-gauss-1step" "01K6X4TBDSBVV7TSXCZECKT79A"
# resume_training "train-era5-n512-e9c1-1step.yaml" "$base_name-era5-n512-e9c1-1step" "01K6DR1108GVPXVHEBSC0DWYZJ"
resume_training "train-era5-n512-e9c1-1step.yaml" "$base_name-era5-n512-e9c1-1step" "01K702HRVH17011GBF0HPNDYT2"

