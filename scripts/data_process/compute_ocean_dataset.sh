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

source activate ocean_data_proc

# NOTE: currently only handle a single run directory and ignore others
run_directory=$(yq -r '.runs | to_entries | .[0].value' ${CONFIG})
run_name=$(yq -r '.runs | to_entries | .[0].key' ${CONFIG})
data_output_directory=$(yq -r '.data_output_directory' ${CONFIG})
output_store="${data_output_directory}/${run_name}.zarr"

if [[ "${COMPUTE_DATASET}" == "true" ]]
then
    python3 -u compute_ocean_dataset.py \
        --config="${CONFIG}" \
        --output-store="${output_store}" \
        --run-directory="${run_directory}"
fi

python3 -u get_stats.py "${CONFIG}" 0
