#!/bin/bash

set -e

ZARR=gs://vcm-ml-intermediate/2023-05-10-vertically-resolved-1deg-fme-dataset.zarr
OUTPUT_DIR="/tmp/beaker-netcdfs/$(uuidgen)"

# save 2010-2019 (intended for training) data netCDFs and create beaker dataset
python convert_to_monthly_netcdf.py \
    $ZARR \
    $OUTPUT_DIR \
    --start-date 2010-01-01 \
    --end-date 2019-12-31

beaker dataset create \
    $OUTPUT_DIR \
    --name "fv3gfs-vertically-resolved-v1-2010-2019" \
    --desc "10 years of data for 8-level vertically resolved model, allowing moisture and dry air mass constraints."

rm -r $OUTPUT_DIR

# save 2020 (intended for validation) data netCDFs and create beaker dataset
python convert_to_monthly_netcdf.py \
    $ZARR \
    $OUTPUT_DIR \
    --start-date 2020-01-01 \
    --end-date 2020-12-31

beaker dataset create \
    $OUTPUT_DIR \
    --name "fv3gfs-vertically-resolved-v1-2020" \
    --desc "1 year of data for 8-level vertically resolved model, allowing moisture and dry air mass constraints."

rm -r $OUTPUT_DIR
