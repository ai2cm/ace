#!/bin/bash
# Submit a CPU beaker job that coarsens the 6-hourly 4deg E3SMv3 piControl
# atmosphere dataset to daily (scripts/data_process/time_coarsen.py) and then
# appends the constant global_mean_co2 (add_constant_co2.py).
#
# Refuses to submit if the output store already exists on GCS:
# time_coarsen.py's existence check only works on local paths.
#
# Usage (from anywhere in the repo): bash submit_daily_coarsen.sh [--dry-run]

set -e

CONFIG=scripts/data_process/configs/E3SMv3-piControl-atmosphere-4deg-8layer-105yr.yaml
OUTPUT=gs://vcm-ml-intermediate/2026-10-22-E3SMv3-piControl-atmosphere-105yr-4deg-daily/2026-10-22-E3SMv3-piControl-atmosphere-105yr-4deg-daily.zarr
GCP_CREDS=/tmp/google_application_credentials.json
JOB_NAME=e3sm-piControl-4deg-daily-coarsen

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
SCRIPT_DIR=configs/experiments/2026-10-19-fm/e3sm

if gsutil ls "$OUTPUT/" >/dev/null 2>&1; then
    echo "Error: $OUTPUT already exists; refusing to overwrite it."
    exit 1
fi

cmd=(gantry run
    --name "$JOB_NAME"
    --task-name "$JOB_NAME"
    --description "Coarsen 6-hourly 4deg E3SMv3 piControl atmosphere to daily and add constant CO2"
    --beaker-image "$(cat latest_deps_only_image.txt)"
    --workspace ai2/ace
    --priority high
    --preemptible
    --budget ai2/atec-climate
    --cluster ai2/phobos
    --gpus 0
    --shared-memory 64GiB
    --env "GOOGLE_APPLICATION_CREDENTIALS=$GCP_CREDS"
    --dataset-secret "google-credentials:$GCP_CREDS"
    --system-python
    --install "pip install 'dask[array]' distributed 'xpartition>=2025.3.0' && pip install --no-deps ."
    --allow-dirty
    -- bash -c "python scripts/data_process/time_coarsen.py $CONFIG 0 && python $SCRIPT_DIR/add_constant_co2.py $OUTPUT")

if [[ "$1" == "--dry-run" ]]; then
    echo "${cmd[@]}"
    exit 0
fi
"${cmd[@]}"
