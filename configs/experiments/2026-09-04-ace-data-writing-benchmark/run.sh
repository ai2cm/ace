#!/bin/bash

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
REPO_ROOT=$(git rev-parse --show-toplevel)
COMMIT=$(git rev-parse --short HEAD)

cd "$REPO_ROOT"

run_arm() {
    local arm="$1"
    local config_filename="$2"
    local job_name="ace-data-writing-benchmark-${arm}-${COMMIT}"

    gantry run \
        --name "$job_name" \
        --description 'Benchmark ACE data writing' \
        --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
        --workspace ai2/ace \
        --priority normal \
        --min-runtime 30m \
        --cluster ai2/phobos \
        --env WANDB_USERNAME="$WANDB_USERNAME" \
        --env WANDB_NAME="$job_name" \
        --env WANDB_JOB_TYPE=benchmark \
        --env WANDB_PROJECT=ace-data-writing-benchmark \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --shared-memory 400GiB \
        --weka climate-default:/climate-default \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- python3 -m fme.ace.inference.data_writer.benchmark "$SCRIPT_PATH/$config_filename"
}

run_arm weka config-weka.yaml
run_arm gcs config-gcs.yaml
