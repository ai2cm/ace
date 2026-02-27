#!/bin/bash

# Options,
# DirectRunner - local
# DataflowRunner - cloud
RUNNER=${1:-DirectRunner}

python3 xr-beam-pipeline.py \
    gs://vcm-ml-scratch/oliwm/test-era5-pipeline/2026-02-25-era5-1deg-8layer-1940-2022.zarr \
    2022-12-26T12:00:00 \
    2022-12-31T18:00:00 \
    --output_grid F90 \
    --output_time_chunksize 2 \
    --output_time_shardsize 12 \
    --process_time_chunksize 4 \
    --project vcm-ml \
    --region us-central1 \
    --temp_location gs://vcm-ml-scratch/oliwm/temp/ \
    --experiments use_runner_v2 \
    --runner $RUNNER \
    --sdk_location container \
    --sdk_container_image us-central1-docker.pkg.dev/vcm-ml/full-model/era5-ingest-dataflow:2025-10-07-era5-xarray-beam-pipelines \
    --save_main_session \
    --num_workers 1 \
    --disk_size_gb 70 \
    --max_num_workers 750 \
    --machine_type n2d-custom-2-24576-ext \
    --worker_disk_type "compute.googleapis.com/projects/vcm-ml/zones/us-central1-c/diskTypes/pd-ssd" \
    --number_of_worker_harness_threads 1
