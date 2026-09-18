#!/bin/bash
# Submit create_coupled_datasets.py to argo via
# create_coupled_datasets_argo_workflow.yaml. Every module in the entry point's
# transitive sibling-import closure is passed as a workflow parameter.

set -e

CONFIG=
IMAGE=
DEBUG=false
SUBSAMPLE=false
DRY_RUN=false

while [[ "$#" -gt 0 ]]
do case $1 in
    --config) CONFIG="$2"
    shift;;
    --image) IMAGE="$2"
    shift;;
    --debug) DEBUG=true;;
    --subsample) SUBSAMPLE=true;;
    --dry-run) DRY_RUN=true;;
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

args=(create_coupled_datasets_argo_workflow.yaml
    -p create_coupled_datasets_script="$(< create_coupled_datasets.py)"
    -p coupled_dataset_utils_script="$(< coupled_dataset_utils.py)"
    -p create_window_avg_dataset_script="$(< create_window_avg_dataset.py)"
    -p time_utils_script="$(< time_utils.py)"
    -p get_stats_script="$(< get_stats.py)"
    -p merge_stats_script="$(< merge_stats.py)"
    -p combine_stats_script="$(< combine_stats.py)"
    -p writer_utils_script="$(< writer_utils.py)"
    -p fs_utils_script="$(< fs_utils.py)"
    -p config="$(< "${CONFIG}")"
    -p debug="${DEBUG}"
    -p subsample="${SUBSAMPLE}")

# --image is omitted when unset, so the workflow's default image applies.
if [[ -n "${IMAGE}" ]]
then
    args+=(-p image="${IMAGE}")
fi

if [[ "${DRY_RUN}" = true ]]
then
    echo "argo submit ${args[*]}"
    exit 0
fi

# Capture the output of the argo submit command
output=$(argo submit "${args[@]}")

# Extract the job name from the output
job_name=$(echo "$output" | grep 'Name:' | awk '{print $2}')

# Print the job name
echo "Argo job submitted: $job_name"
