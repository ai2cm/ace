#!/bin/bash

set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT

BEAKER_IMAGE=spencerc/hf-cli-gantry
JOB_NAME=hf-sync-example

STORE=abrupt4xCO2-ic_0001.zarr
SOURCE=/climate-default/2025-02-07-vertically-resolved-1deg-c96-shield-som-abrupt-4xCO2-ensemble-fme-dataset/${STORE}
DESTINATION=hf://buckets/allenai/ai2cm-scratch/abrupt-4xCO2-ensemble/${STORE}

# HF_XET_HIGH_PERFORMANCE maximizes upload parallelism (see Dockerfile), but for
# stores containing multiple large (>4GB) files this can cause upload timeouts.
# Set to 0 to disable it in that case. Overrides the default baked into the image.
HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}

# Set to 1 to pass --ignore-existing, which skips files already present at the
# destination rather than re-uploading them. Useful for resuming a failed sync so
# it doesn't redo completed work. Left off by default so a fresh run performs a
# full sync (uploading new files and re-uploading any that changed at the source).
IGNORE_EXISTING=${IGNORE_EXISTING:-0}
SYNC_ARGS=()
if [ "${IGNORE_EXISTING}" = "1" ]; then
    SYNC_ARGS+=(--ignore-existing)
fi

gantry run \
    --name "${JOB_NAME}" \
    --description 'Sync dataset on WEKA with a Hugging Face bucket' \
    --beaker-image "${BEAKER_IMAGE}" \
    --workspace ai2/ace \
    --priority high \
    --cluster ai2/phobos \
    --env-secret HF_TOKEN=hugging-face-token \
    --env HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE}" \
    --gpus 0 \
    --shared-memory 64GiB \
    --min-runtime 8h \
    --no-python \
    --allow-dirty \
    --weka climate-default:/climate-default \
    -- hf sync "${SYNC_ARGS[@]}" "${SOURCE}" "${DESTINATION}"
