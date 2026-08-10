#!/bin/bash

set -euo pipefail

# Space-separated list; inference runs all ICs of a config as a single batch, so
# a large IC window must be split across configs (and therefore across jobs).
CONFIG_FILENAMES="${CONFIG_FILENAMES:-${CONFIG_FILENAME:-evaluate-coupled-nino-scalars.yaml}}"
JOB_NAME="${JOB_NAME:-cm4-coupled-nino-oos-one-step}"
JOB_GROUP="${JOB_GROUP:-cm4-1pct-samudra-nino}"
COUPLED_RESULTS_DATASET="${COUPLED_RESULTS_DATASET:-01KY3DATM3CAEA479JQZQDPT9W}"
COUPLED_CKPT="${COUPLED_CKPT:-best_inference_ckpt}"

REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
COMPACT_SCRIPT="${SCRIPT_PATH}/compact_nino_scalar_forecasts.py"
# Single GPU: raw netCDF writers are not multi-rank safe (all ranks open the
# same /results path). Matches evaluate-nino-scalars.sh.
N_GPUS=1

cd "$REPO_ROOT"

read -r -a config_filenames <<< "$CONFIG_FILENAMES"

# Gantry runs the pushed commit, not the working tree, so a file that exists
# locally but is uncommitted or unpushed is simply absent on the worker. Fail
# here instead of after the job has queued and started.
check_shipped() {
    local path=$1
    if ! git cat-file -e "HEAD:$path" 2>/dev/null; then
        echo "ERROR: $path is not committed at HEAD; the worker will not see it." >&2
        exit 1
    fi
    if ! git diff --quiet HEAD -- "$path"; then
        echo "ERROR: $path has uncommitted changes that will not be shipped." >&2
        exit 1
    fi
}

if [[ -z $(git branch -r --contains HEAD 2>/dev/null) ]]; then
    echo "ERROR: HEAD is not on any remote branch; push before launching." >&2
    exit 1
fi

check_shipped "$COMPACT_SCRIPT"
for config_filename in "${config_filenames[@]}"; do
    check_shipped "${SCRIPT_PATH}/${config_filename}"
    python -m fme.coupled.validate_config --config_type evaluator \
        "${SCRIPT_PATH}/${config_filename}"
done

BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')

run_eval() {
    local config_filename=$1
    local job_name=$2
    local config_path="${SCRIPT_PATH}/${config_filename}"
    local n_ics
    n_ics=$(python -c \
        "import sys, yaml; print(len(yaml.safe_load(open(sys.argv[1]))['loader']['start_indices']['times']))" \
        "$config_path")

    gantry run \
        --name "$job_name" \
        --task-name "$job_name" \
        --description "${n_ics} out-of-sample one-step coupled FT Nino3.4 scalar forecasts" \
        --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
        --workspace ai2/climate-titan \
        --priority high \
        --preemptible \
        --cluster ai2/titan \
        --weka climate-default:/climate-default \
        --env WANDB_USERNAME="$BEAKER_USERNAME" \
        --env WANDB_NAME="$job_name" \
        --env WANDB_JOB_TYPE=evaluation \
        --env WANDB_RUN_GROUP="$JOB_GROUP" \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset "$COUPLED_RESULTS_DATASET:training_checkpoints/${COUPLED_CKPT}.tar:/ckpt.tar" \
        --gpus "$N_GPUS" \
        --shared-memory 400GiB \
        --budget ai2/atec-climate \
        --allow-dirty \
        --system-python \
        --install "pip install --no-deps ." \
        -- bash -c \
            "python -m fme.coupled.evaluator '$config_path' && \
             python '$COMPACT_SCRIPT' \
               --input-dir /results/raw/ocean \
               --output-dir /results/nino_scalar_forecasts \
               --checkpoint-dataset '$COUPLED_RESULTS_DATASET' \
               --description 'Direct Nino3.4 scalar forecasts from one coupled FT ocean step over CM4 1pctCO2 ICs matching the ocean-only OOS eval.'"
}

for config_filename in "${config_filenames[@]}"; do
    if [[ ${#config_filenames[@]} -eq 1 ]]; then
        job_name="$JOB_NAME"
    else
        # Suffix the job with the config's chunk tag so jobs stay distinguishable.
        chunk_tag=${config_filename%.yaml}
        chunk_tag=${chunk_tag##*-}
        job_name="${JOB_NAME}-${chunk_tag}"
    fi
    run_eval "$config_filename" "$job_name"
done
