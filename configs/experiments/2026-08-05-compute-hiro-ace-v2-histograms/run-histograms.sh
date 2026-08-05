#!/bin/bash

set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

HISTOGRAM_SCRIPT="scripts/downscaling/histogram.py"
ZARR_PATH="/climate-default/2026-08-03-hiro-ace-v2-outputs/2026-08-03-hiro-perfect-pred-global-2023/global_perfect_prediction.zarr"
OUTPUT_DIR="/climate-default/2026-08-03-hiro-ace-v2-outputs/2026-08-03-hiro-perfect-pred-global-2023/prediction_histograms"

submit_histogram_job() {
    local job_name="$1"
    shift
    local variables=("$@")

    gantry run \
        --name "$job_name" \
        --description "Compute histograms over 1 year of global 2023 X-SHiELD" \
        --workspace ai2/ace \
        --priority normal \
        --cluster ai2/phobos \
        --gpus 0 \
        --budget ai2/atec-climate \
        --beaker-image "$DEPS_ONLY_IMAGE" \
        --weka climate-default:/climate-default \
        --system-python \
        --allow-dirty \
        --install "pip install --no-deps ." \
        -- bash -c "pip install 'dask[array]' && python '$HISTOGRAM_SCRIPT' '$ZARR_PATH' \
            --progress-interval 120 \
            --output-dir '$OUTPUT_DIR' \
            --lat-range -65 65 \
            --variables ${variables[*]}"
}

submit_histogram_job "hiro-histograms-precip-mslp-perfect-pred-2023" PRATEsfc PRMSL
submit_histogram_job "hiro-histograms-wind-perfect-pred-2023" UGRD10m VGRD10m wind_speed
