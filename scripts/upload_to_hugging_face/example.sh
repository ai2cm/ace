#!/bin/bash

set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT

N_GPUS=0
BEAKER_IMAGE=spencerc/hf-cli-gantry
JOB_NAME=hf-sync-example

STORE=abrupt4xCO2-ic_0001.zarr
SOURCE=/climate-default/2025-02-07-vertically-resolved-1deg-c96-shield-som-abrupt-4xCO2-ensemble-fme-dataset/${STORE}
DESTINATION=hf://buckets/allenai/ai2cm-scratch/abrupt-4xCO2-ensemble/${STORE}

gantry run \
    --name "${JOB_NAME}" \
    --description 'Sync dataset on WEKA with a Hugging Face bucket' \
    --beaker-image "${BEAKER_IMAGE}" \
    --workspace ai2/ace \
    --priority high \
    --cluster ai2/phobos \
    --env-secret HF_TOKEN=hugging-face-token \
    --gpus "${N_GPUS}" \
    --shared-memory 64GiB \
    --min-runtime 8h \
    --no-python \
    --allow-dirty \
    --weka climate-default:/climate-default \
    -- hf sync "${SOURCE}" "${DESTINATION}"
