#!/bin/bash

# Options,
# DirectRunner - local
# PortableRunner - container test
# DataflowRunner - cloud
RUNNER=${1:-DataflowRunner}

python3 xr-beam-pipeline.py \
    gs://vcm-ml-intermediate/2024-04-15-era5-1deg-8layer.zarr \
    100 \
    --ncar_process_time_chunksize 10 \
    --project vcm-ml \
    --region us-central1 \
    --temp_location gs://vcm-ml-scratch/oliwm/temp/ \
    --experiments use_runner_v2 \
    --runner $RUNNER \
    --sdk_location container \
    --sdk_container_image us.gcr.io/vcm-ml/era5-ingest-dataflow:2024-03-11-era5-xarray-beam-pipelines \
    --save_main_session \
    --num_workers 1 \
    --disk_size_gb 70 \
    --max_num_workers 750 \
    --machine_type n2d-custom-2-24576-ext \
    --worker_disk_type "compute.googleapis.com/projects/vcm-ml/zones/us-central1-c/diskTypes/pd-ssd" \
    --number_of_worker_harness_threads 1


# save_main_session is needed so that the imported modules are available to individual functions
# disk_size_gb should be large enough to host a single timestep converted into grib
# There might be some issues with clearing the metview cache as I did notice some disk full
# messages on the scaled pipeline