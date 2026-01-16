#!/bin/bash

python get_sample_xshield_data.py \
    --input-url gs://vcm-ml-intermediate/2025-09-16-X-SHiELD-AMIP-1deg-8layer-11yr.zarr  \
    --forcing-start-time 2014 \
    --forcing-end-time 2024 \
    --compress-forcing \
    ./data/sample_xshield_data
