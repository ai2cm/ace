#!/bin/bash

# Options,
# DirectRunner - run pipeline locally, good for testing and debugging
# DataflowRunner - run pipeline on Google Cloud Dataflow, good for large scale processing
RUNNER="${1}"
OUTPUT_PATH="${2}"
OUTPUT_GRID="${3}"
START_TIME="${4}"
END_TIME="${5}"
EXTRA_FLAGS=("${@:6}")

# Common args shared by all runners
COMMON_ARGS=(
    $OUTPUT_PATH
    $START_TIME
    $END_TIME
    --output_grid $OUTPUT_GRID
    --output_time_chunksize 1
    --output_time_shardsize 240
    --process_time_chunksize 2
    --runner="$RUNNER"
    --save_main_session
)

# DirectRunner-specific args. Without this, got a gRPC timeout error.
DIRECT_ARGS=(
    --job_server_timeout=3600
)

# Dataflow-specific args (gRPC runner, cloud resources, container image)
DATAFLOW_ARGS=(
    --project vcm-ml
    --region us-central1
    --temp_location gs://vcm-ml-scratch/era5-ingestion-pipeline/temp/
    --experiments use_runner_v2
    --sdk_location container
    --sdk_container_image us-central1-docker.pkg.dev/vcm-ml/full-model/era5-ingest-dataflow:2026-03-03-era5-xarray-beam-pipelines
    --num_workers 1
    --disk_size_gb 70
    --max_num_workers 750
    --machine_type n2d-custom-2-49152-ext
    --worker_disk_type "compute.googleapis.com/projects/vcm-ml/zones/us-central1-c/diskTypes/pd-ssd"
    --number_of_worker_harness_threads 1
)

if [ "$RUNNER" = "DataflowRunner" ]; then
    python3 xr-beam-pipeline.py "${COMMON_ARGS[@]}" "${EXTRA_FLAGS[@]}" "${DATAFLOW_ARGS[@]}"
else
    python3 xr-beam-pipeline.py "${COMMON_ARGS[@]}" "${EXTRA_FLAGS[@]}" "${DIRECT_ARGS[@]}"
fi
