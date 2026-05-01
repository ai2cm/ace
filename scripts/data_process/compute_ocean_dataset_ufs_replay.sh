#!/bin/bash

set -e

COMPUTE_DATASET=true

while [[ "$#" -gt 0 ]]
do case $1 in
    --config) CONFIG="$2"
    shift;;
    --stats-only) COMPUTE_DATASET=false;;
    *) echo "Unknown parameter passed: $1"
    exit 1;;
esac
shift
done


if [[ -z "${CONFIG}" ]]
then
    echo "Option --config missing"
    exit 1;
fi

echo "full-model SHA: $(git rev-parse HEAD)"

# NOTE: currently only handle a single output store
output_directory=$(yq -r '.output_directory' ${CONFIG})
output_store="${output_directory}/ufs-replay-ocean-1deg.zarr"

if [[ "${COMPUTE_DATASET}" == "true" ]]
then
    python3 -u compute_ocean_dataset_ufs_replay.py \
        --config="${CONFIG}" \
        --output-store="${output_store}"
fi

# Compute stats on the output zarr
# Uses get_stats.py with a temporary config pointing to the output store
python3 -u -c "
import xarray as xr
import os, sys
sys.path.insert(0, os.path.dirname('$0'))
from get_stats import get_stats, StatsConfig

config = StatsConfig(
    output_directory='${output_directory}/stats',
    data_type='UFS_REPLAY',
)
get_stats(config, '${output_store}', '${output_directory}/stats', debug=False)
"
