#!/bin/bash


# The dependencies of this script are installed in the "fv3net" conda environment
# which can be installed using fv3net's Makefile. See
# https://github.com/ai2cm/fv3net/blob/8ed295cf0b8ca49e24ae5d6dd00f57e8b30169ac/Makefile#L310

# This configuration of the script was run on a GCP VM with 8 workers, and took around
# ~10 hours to complete.  The roundtrip_filter adds significant computational time.
# If we plan to do this regularly, paralellizing the script across time using xpartition to
# launch jobs for different time chunks would probably be a good idea.

set -ex

python compute_dataset_fv3gfs.py \
    --root "gs://vcm-ml-raw-flexible-retention/2023-08-03-C48-FME-reference-ensemble/regridded-zarrs/gaussian_grid_180_by_360/ic_{ic:04d}_2021010100" \
    --output "gs://vcm-ml-intermediate/2023-09-01-vertically-resolved-1deg-fme-c48-baseline-dataset/ic_{ic:04d}.zarr" \
    --ic 11 --n-workers 8 --roundtrip-fraction-kept 0.65
