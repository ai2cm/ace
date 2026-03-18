
#!/bin/bash

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=1

cd "$REPO_ROOT"

JOB_NAME="fme-core-benchmarks"

run_benchmark() {
    local cluster="$1"

    if [ -z "$cluster" ]; then
        echo "Usage: run_benchmark <cluster>"
        return 1
    fi

    gantry run \
        --name "$JOB_NAME" \
        --description 'Run ACE training' \
        --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
        --workspace ai2/climate-titan \
        --priority urgent \
        --preemptible \
        --cluster "$cluster" \
        --env WANDB_USERNAME="$WANDB_USERNAME" \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --gpus "$N_GPUS" \
        --shared-memory 400GiB \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --allow-dirty \
        --system-python \
        --install "pip install --no-deps ." \
        -- python3 -m fme.core.benchmark.run \
            --wandb-project ai2cm/fme-core-benchmarks
}

run_benchmark "ai2/titan"
run_benchmark "ai2/jupiter"
