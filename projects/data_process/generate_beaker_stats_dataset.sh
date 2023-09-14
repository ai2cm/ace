#!/bin/bash

set -e

ZARR=gs://vcm-ml-intermediate/2023-08-09-vertically-resolved-1deg-fme-ensemble-dataset/ic_0001.zarr
OUTPUT_DIR="/tmp/$(uuidgen)"

python get_stats.py \
    $ZARR \
    $OUTPUT_DIR \
    --start-date 2021-01-01 \
    --end-date 2030-12-31

beaker dataset create \
    $OUTPUT_DIR \
    --name "fv3gfs-ensemble-ic0001-stats-residual-scaling-all-years-v4" \
    --desc "Coefficients for normalization for 8-level vertically resolved dataset with annually-repeating SSTs, using residual scaling. Includes surface height. Residual scaling only applied to prognostic variables, with min standard deviation set to 1."

rm -r $OUTPUT_DIR

python get_stats.py \
    $ZARR \
    $OUTPUT_DIR \
    --start-date 2021-01-01 \
    --end-date 2030-12-31 \
    --full-field

beaker dataset create \
    $OUTPUT_DIR \
    --name "fv3gfs-ensemble-ic0001-stats-full-field-scaling-all-years-v4" \
    --desc "Coefficients for normalization for 8-level vertically resolved dataset with annually-repeating SSTs, using full field scaling. Includes surface height."

rm -r $OUTPUT_DIR
