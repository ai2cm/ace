#!/bin/bash

rm -r /Users/oliverwm/scratch/sample_era5_data

python get_sample_era5_data.py \
    --input-url gs://vcm-ml-intermediate/2024-06-20-era5-1deg-8layer-1940-2022.zarr \
    --forcing-start-time 1940 \
    --forcing-end-time 2022 \
    --compress-forcing \
    /Users/oliverwm/scratch/sample_era5_data