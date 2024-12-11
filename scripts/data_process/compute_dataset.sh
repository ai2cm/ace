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

names=($(yq -r '.runs | to_entries[].key' ${CONFIG}))
run_directories=($(yq -r '.runs | to_entries[].value' ${CONFIG}))
output_directory=$(yq -r '.data_output_directory' ${CONFIG})
runs_count=$(yq -r '.runs | length' ${CONFIG})
runs_count_minus_one=$(($runs_count - 1))

# Capture the output of the argo submit command
output=$(argo submit compute_dataset_argo_workflow.yaml \
    -p compute_dataset=${COMPUTE_DATASET} \
    -p python_script="$(< compute_dataset.py)" \
    -p get_stats_script="$(< get_stats.py)" \
    -p combine_stats_script="$(< combine_stats.py)" \
    -p upload_stats_script="$(< upload_stats.py)" \
    -p config="$(< ${CONFIG})" \
    -p names="${names[*]}" \
    -p run_directories="${run_directories[*]}" \
    -p output_directory="${output_directory}" \
    -p runs_count_minus_one=${runs_count_minus_one})

# Extract the job name from the output
job_name=$(echo "$output" | grep 'Name:' | awk '{print $2}')

# Print the job name
echo "Argo job submitted: $job_name"
