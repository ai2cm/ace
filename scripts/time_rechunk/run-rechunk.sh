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

# BASE_DIR="gs://vcm-ml-intermediate/2024-06-04-vertically-resolved-4deg-fme-ensemble-dataset"
# DEST_DIR="gs://vcm-ml-intermediate/2024-06-06-vertically-resolved-4deg-fme-ensemble-dataset-5mb-chunks"
# TIME_CHUNK=320
# NUM_IC=11

BASE_DIR=" gs://vcm-ml-intermediate/2024-06-04-vertically-resolved-4deg-fme-amip-ensemble-dataset"
DEST_DIR=" gs://vcm-ml-intermediate/2024-06-04-vertically-resolved-4deg-fme-amip-ensemble-dataset-5mb-chunks"
TIME_CHUNK=320
NUM_IC=4

LOG_DIR="logs/$(date +'%Y%m%d_%H%M%S')"
mkdir -p ${LOG_DIR}

for i in $(seq -w 1 ${NUM_IC}); do
    FILE_NAME=$(printf "ic_%04d.zarr" $i)
    LOG_FILE="${LOG_DIR}/rechunk_${i}.log"
    SRC="${BASE_DIR}/${FILE_NAME}"
    DEST="${DEST_DIR}/${FILE_NAME}"
    python rechunk_zarr.py \
        ${SRC} \
        ${DEST} \
        ${TIME_CHUNK} \
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
        --machine_type n2d-standard-2 \
        >> ${LOG_FILE} 2>&1 &
done

# save_main_session is needed so that the imported modules are available to individual functions
