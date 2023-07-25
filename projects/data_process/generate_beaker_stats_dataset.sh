#!/bin/bash

set -e

ZARR=gs://vcm-ml-intermediate/2023-07-18-vertically-resolved-1deg-fme-ensemble-dataset/ic_0001.zarr
OUTPUT_DIR="/tmp/$(uuidgen)"

python get_stats.py \
    $ZARR \
    $OUTPUT_DIR \
    --start-date 2021-01-01 \
    --end-date 2030-12-31

beaker dataset create \
    $OUTPUT_DIR \
    --name "fv3gfs-ensemble-ic0001-stats-residual-scaling-all-years" \
    --desc "Coefficients for normalization for 8-level vertically resolved dataset with annually-repeating SSTs, using residual scaling."

rm -r $OUTPUT_DIR

python get_stats.py \
    $ZARR \
    $OUTPUT_DIR \
    --start-date 2021-01-01 \
    --end-date 2030-12-31 \
    --full-field

beaker dataset create \
    $OUTPUT_DIR \
    --name "fv3gfs-ensemble-ic0001-stats-full-field-scaling-all-years" \
    --desc "Coefficients for normalization for 8-level vertically resolved dataset with annually-repeating SSTs, using full field scaling."

rm -r $OUTPUT_DIR
