#!/bin/bash

set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT

N_GPUS=0
BEAKER_IMAGE=spencerc/rclone-gantry
JOB_NAME=rclone-copy-example

STORE=abrupt4xCO2-ic_0001.zarr
SOURCE=/climate-default/2025-02-07-vertically-resolved-1deg-c96-shield-som-abrupt-4xCO2-ensemble-fme-dataset/${STORE}
DESTINATION=hf:ai2cm-scratch/abrupt-4xCO2-ensemble/${STORE}

gantry run \
    --name $JOB_NAME \
    --description 'Copy dataset from WEKA to Hugging Face' \
    --beaker-image ${BEAKER_IMAGE} \
    --workspace ai2/ace \
    --priority high \
    --cluster ai2/phobos \
    --dataset-secret hugging-face-credentials:/config/rclone/rclone.conf \
    --gpus $N_GPUS \
    --shared-memory 64GiB \
    --min-runtime 8h \
    --no-python \
    --allow-dirty \
    --weka climate-default:/climate-default \
    -- rclone copy \
        --checksum \
        --transfers 128 \
        --checkers 128 \
        --progress \
        --stats-one-line \
        ${SOURCE} \
        ${DESTINATION}
