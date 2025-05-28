#!/bin/bash

set -e

JOB_NAME="ace-data-benchmark-v2-30-steps-16dw-noopen"  # recommended but not required to change this
CONFIG_PATH=configs/experiments/2025-05-23-ace-data-loading-benchmark/config.yaml  # relative to the root of the repository

BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
WANDB_PROJECT=ace-data-benchmark

gantry run \
    --name $JOB_NAME \
    --description 'Benchmark ACE data loading' \
    --beaker-image jeremym/fme-deps-only-5a6649b8 \
    --workspace ai2/ace \
    --priority normal \
    --cluster ai2/rhea-cirrascale \
    --shared-memory 200GiB \
    --env WANDB_USERNAME=$WANDB_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=benchmark \
    --env WANDB_PROJECT=$WANDB_PROJECT \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --weka climate-default:/climate-default \
    --budget ai2/climate \
    --no-conda \
    --install "pip install --no-deps ." \
    -- python -m fme.ace.data_loading.benchmark $CONFIG_PATH
