#!/bin/bash

set -e

ZARR=gs://vcm-ml-intermediate/2023-05-10-vertically-resolved-1deg-fme-dataset.zarr
OUTPUT_DIR="/tmp/$(uuidgen)"

python get_stats_fv3gfs.py \
    $ZARR \
    $OUTPUT_DIR \
    --start-date 2010-01-01 \
    --end-date 2020-12-31

beaker dataset create \
    $OUTPUT_DIR \
    --name "fv3gfs-vertically-resolved-v1-stats-residual-scaling-all-years" \
    --desc "Coefficients for normalization for 8-level vertically resolved dataset, using residual scaling."

rm -r $OUTPUT_DIR

python get_stats_fv3gfs.py \
    $ZARR \
    $OUTPUT_DIR \
    --start-date 2010-01-01 \
    --end-date 2020-12-31 \
    --full-field

beaker dataset create \
    $OUTPUT_DIR \
    --name "fv3gfs-vertically-resolved-v1-stats-full-field-scaling-all-years" \
    --desc "Coefficients for normalization for 8-level vertically resolved dataset, using full field scaling."

rm -r $OUTPUT_DIR
