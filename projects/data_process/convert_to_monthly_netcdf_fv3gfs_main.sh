#!/bin/bash

# This script launches 11 jobs to convert all GCS zarr data to local monthly netCDFs
# and imposes a train/validation data split.

BASE_URL=gs://vcm-ml-intermediate/2023-08-09-vertically-resolved-1deg-fme-ensemble-dataset

for IC in "0001" "0002" "0003" "0004" "0005" "0006" "0007" "0008" "0009" "0010"; do
    INPUT_URL=${BASE_URL}/ic_${IC}.zarr
    python convert_to_monthly_netcdf.py \
        $INPUT_URL \
        /net/nfs/climate/data/2023-08-11-vertically-resolved-1deg-fme-ensemble-dataset/train/ic_${IC} \
        --start-date 2021-01-01 \
        --end-date 2030-12-31 &
done

INPUT_URL=${BASE_URL}/ic_0011.zarr
python convert_to_monthly_netcdf.py \
    $INPUT_URL \
    /net/nfs/climate/data/2023-08-11-vertically-resolved-1deg-fme-ensemble-dataset/validation/ic_0011 \
    --start-date 2021-01-01 \
    --end-date 2030-12-31 &

# process C48 baseline run (with roundtrip filter truncation, 0.65 mode fraction kept)
python convert_to_monthly_netcdf.py \
    --prepend-nans \
    gs://vcm-ml-intermediate/2023-09-01-vertically-resolved-1deg-fme-c48-baseline-dataset/ic_0011.zarr \
    /net/nfs/climate/data/2023-09-12-vertically-resolved-1deg-fme-c48-baseline-dataset-truncated-065/ic_0011 \
    --start-date 2021-01-01 \
    --end-date 2030-12-31 &
