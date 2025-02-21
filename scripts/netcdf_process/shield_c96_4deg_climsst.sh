#!/bin/bash

set -e

JOB_NAME="shield-c96-4deg-climsst"
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT && gantry run \
    --name $JOB_NAME \
    --description 'Convert 4-degree C96 FV3GFS climSST ensemble to monthly netcdfs' \
    --beaker-image oliverwm/fme-deps-only-2025-01-16 \
    --workspace ai2/ace \
    --priority normal \
    --cluster ai2/phobos-cirrascale \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --gpus 0 \
    --shared-memory 40GiB \
    --weka climate-default:/climate-default \
    --budget ai2/climate \
    --no-conda \
    --install "pip install --no-deps ." \
    -- bash -c "cd /gantry-runtime/scripts/data_process && make fv3gfs_4deg_climSST_monthly_netcdfs"
