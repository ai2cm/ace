#!/bin/bash

# Options:
# DirectRunner - run pipeline locally, good for testing and debugging
# DataflowRunner - run pipeline on Google Cloud Dataflow
RUNNER="${1}"
OUTPUT_PATH="${2}"
OUTPUT_GRID="${3}"  # F90
START_DATE="${4}"
END_DATE="${5}"
EXTRA_FLAGS=("${@:6}")

# Common args shared by all runners
COMMON_ARGS=(
    $OUTPUT_PATH
    $START_DATE
    $END_DATE
    --output_grid $OUTPUT_GRID
    --output_time_chunksize 1
    --output_time_shardsize 360
    --time_stride 5
    --runner="$RUNNER"
    --save_main_session
)

# DirectRunner-specific args
DIRECT_ARGS=(
    --job_server_timeout=3600
)

# Dataflow-specific args
DATAFLOW_ARGS=(
    --project vcm-ml
    --region us-central1
    --temp_location gs://vcm-ml-scratch/glorys-pipeline/temp/
    --experiments use_runner_v2
    --sdk_location container
    --sdk_container_image us-central1-docker.pkg.dev/vcm-ml/full-model/glorys-ingest-dataflow:latest
    --num_workers 1
    --disk_size_gb 100
    # The binding constraint is CloudFerro egress, not compute: a single worker
    # sustained ~40 MB/s on the test run. Workers past the point where their
    # aggregate saturates the source sit idle waiting on I/O and are billed
    # anyway, so autoscaling far beyond it buys no wall time and multiplies
    # cost. Raise only after confirming per-worker throughput holds.
    --max_num_workers ${MAX_NUM_WORKERS:-20}
    --machine_type n2d-custom-2-49152-ext
    --worker_disk_type "compute.googleapis.com/projects/vcm-ml/zones/us-central1-c/diskTypes/pd-ssd"
    --number_of_worker_harness_threads 1
)

if [ "$RUNNER" = "DataflowRunner" ]; then
    python glorys-pipeline.py "${COMMON_ARGS[@]}" "${EXTRA_FLAGS[@]}" "${DATAFLOW_ARGS[@]}"
else
    python glorys-pipeline.py "${COMMON_ARGS[@]}" "${EXTRA_FLAGS[@]}" "${DIRECT_ARGS[@]}"
fi
