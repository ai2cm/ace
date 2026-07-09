#!/bin/bash
set -e

# Options,
# DirectRunner - run pipeline locally, good for testing and debugging
# DataflowRunner - run pipeline on Google Cloud Dataflow, good for large scale processing
RUNNER="${1}"
CONFIG="${2}"
EXTRA_FLAGS=("${@:3}")

if [ "$RUNNER" = "DataflowRunner" ]; then
    : "${SDK_CONTAINER_IMAGE:?set SDK_CONTAINER_IMAGE to the worker image URL}"
fi

# Common args shared by all runners. Extra pipeline.run flags (e.g.
# --num-timesteps, --output-path) and beam pipeline options can be appended
# after the config argument.
COMMON_ARGS=(
    --config "$CONFIG"
    --runner="$RUNNER"
)

# DirectRunner-specific args. Without this, got a gRPC timeout error.
DIRECT_ARGS=(
    --job_server_timeout=3600
)

# Dataflow-specific args (gRPC runner, cloud resources, container image).
# Workers use the project service account for GCS access; no S3/OSN
# credentials are present or needed.
DATAFLOW_ARGS=(
    --project vcm-ml
    --region us-central1
    --temp_location gs://vcm-ml-scratch/jamesd/gfdl-om4-ingestion/temp/
    --experiments use_runner_v2
    --sdk_location container
    --sdk_container_image "$SDK_CONTAINER_IMAGE"
    --num_workers 1
    --disk_size_gb 70
    --max_num_workers 100
    --machine_type n2d-custom-2-49152-ext
    --worker_disk_type "compute.googleapis.com/projects/vcm-ml/zones/us-central1-c/diskTypes/pd-ssd"
    --number_of_worker_harness_threads 1
)

if [ "$RUNNER" = "DataflowRunner" ]; then
    python -m pipeline.run "${COMMON_ARGS[@]}" "${EXTRA_FLAGS[@]}" "${DATAFLOW_ARGS[@]}"
else
    python -m pipeline.run "${COMMON_ARGS[@]}" "${EXTRA_FLAGS[@]}" "${DIRECT_ARGS[@]}"
fi
