#!/bin/bash

# Options,
# DirectRunner - local
# DataflowRunner - cloud
RUNNER=${1:-DataflowRunner}
#RUNNER=${1:-DirectRunner}

# Splits each of the original 8 layers into half at linear pressure midpoint
OUTPUT_LAYER_INDICES_16='0 38 48 59 67 74 79 85 90 95 100 104 109 113 119 125 137'
OUTPUT_PATH='gs://vcm-ml-intermediate/2024-07-11-era5-1deg-16layer-1940-2022.zarr'

python3 xr-beam-pipeline.py \
    $OUTPUT_PATH \
    1940-01-01T12:00:00 \
    2022-12-31T18:00:00 \
    --output-layer-indices $OUTPUT_LAYER_INDICES_16 \
    --output_grid F90 \
    --output_time_chunksize 20 \
    --ncar_process_time_chunksize 4 \
    --project vcm-ml \
    --region us-central1 \
    --temp_location gs://vcm-ml-scratch/annak/temp \
    --experiments use_runner_v2 \
    --runner $RUNNER \
    --sdk_location container \
    --sdk_container_image us.gcr.io/vcm-ml/era5-ingest-dataflow:2024-07-10-era5-xarray-beam-pipelines \
    --save_main_session \
    --num_workers 1 \
    --disk_size_gb 70 \
    --max_num_workers 750 \
    --machine_type n2d-custom-2-24576-ext \
    --worker_disk_type "compute.googleapis.com/projects/vcm-ml/zones/us-central1-c/diskTypes/pd-ssd" \
    --number_of_worker_harness_threads 1