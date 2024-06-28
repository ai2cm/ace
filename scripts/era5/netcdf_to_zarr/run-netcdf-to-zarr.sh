#!/bin/bash

# Options,
# DirectRunner - local
# PortableRunner - container test
# DataflowRunner - cloud
RUNNER=${1}
IMAGE_NAME=${2}

for category in e5.oper.fc.sfc.meanflux e5.oper.an.sfc e5.oper.invariant
do
    python netcdf_to_zarr_pipeline.py \
        gs://vcm-ml-intermediate/2024-05-17-era5-025deg-2D-variables-from-NCAR-as-zarr \
        ${category} \
        --project vcm-ml \
        --region us-central1 \
        --temp_location gs://vcm-ml-scratch/oliverwm/temp/ \
        --experiments use_runner_v2 \
        --runner ${RUNNER} \
        --sdk_location container \
        --sdk_container_image ${IMAGE_NAME} \
        --save_main_session \
        --num_workers 1 \
        --disk_size_gb 35 \
        --machine_type n2d-highmem-2
done

# save_main_session is needed so that the imported modules are available to individual functions
