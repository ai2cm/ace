#!/bin/bash

# Options,
# DirectRunner - local
# PortableRunner - container test
# DataflowRunner - cloud

if [ -z "$1" ]; then
    echo "Error: Please provide a runner as the first argument."
    exit 1
fi
RUNNER=${1}
IMAGE_NAME=us.gcr.io/vcm-ml/era5-ingest-dataflow:2024-03-11-era5-xarray-beam-pipelines

BASE_PATH="gs://vcm-ml-intermediate/2024-06-20-era5-1deg-8layer-1940-2022.zarr"
DEST_PATH="gs://vcm-ml-scratch/andrep/chunk-160/2024-06-20-era5-1deg-8layer-1940-2022.zarr"
TIME_CHUNK=160

python rechunk_zarr.py \
    ${BASE_PATH} \
    ${DEST_PATH} \
    ${TIME_CHUNK} \
    --era5 \
    --project vcm-ml \
    --region us-central1 \
    --temp_location gs://vcm-ml-scratch/andrep/temp/ \
    --experiments use_runner_v2 \
    --runner ${RUNNER} \
    --sdk_location container \
    --sdk_container_image ${IMAGE_NAME} \
    --save_main_session \
    --num_workers 1 \
    --disk_size_gb 35 \
    --machine_type n2d-standard-2