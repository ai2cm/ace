#!/bin/bash

# Options,
# DirectRunner - run pipeline locally, good for testing and debugging
# DataflowRunner - run pipeline on Google Cloud Dataflow, good for large scale processing
RUNNER=${1:-DirectRunner}

# Common args shared by all runners
COMMON_ARGS=(
    gs://vcm-ml-scratch/oliwm/test-updated-era5-pipeline/era5-1deg-8layer-1940-2022.zarr
    2022-12-30T00:00:00
    2022-12-31T18:00:00
    --output_grid F90
    --output_time_chunksize 2
    --output_time_shardsize 12
    --process_time_chunksize 2
    --runner="$RUNNER"
    --save_main_session
)

# DirectRunner-specific args. Without this, got a gRCP timeout error.
DIRECT_ARGS=(
    --job_server_timeout=3600
)

# Dataflow-specific args (gRPC runner, cloud resources, container image)
DATAFLOW_ARGS=(
    --project vcm-ml
    --region us-central1
    --temp_location gs://vcm-ml-scratch/oliwm/temp/
    --experiments use_runner_v2
    --sdk_location container
    --sdk_container_image us-central1-docker.pkg.dev/vcm-ml/full-model/era5-ingest-dataflow:2025-10-07-era5-xarray-beam-pipelines
    --num_workers 1
    --disk_size_gb 70
    --max_num_workers 750
    --machine_type n2d-custom-2-24576-ext
    --worker_disk_type "compute.googleapis.com/projects/vcm-ml/zones/us-central1-c/diskTypes/pd-ssd"
    --number_of_worker_harness_threads 1
)

if [ "$RUNNER" = "DataflowRunner" ]; then
    python3 xr-beam-pipeline.py "${COMMON_ARGS[@]}" "${DATAFLOW_ARGS[@]}"
else
    python3 xr-beam-pipeline.py "${COMMON_ARGS[@]}" "${DIRECT_ARGS[@]}"
fi
