#!/bin/bash
# Submit create_coupled_datasets.py to argo via
# create_coupled_datasets_argo_workflow.yaml, which then uploads the merged
# stats to the config's stats.beaker_dataset with upload_coupled_stats.py.
# --dependent-config names a config that reads the --config outputs (e.g.
# 1pctCO2's precomputed_sea_ice_mask reads the piControl coupled ocean store);
# the same workflow runs it once the --config datasets are written. Every
# module in the two entry points' transitive sibling-import closures is passed
# as a workflow parameter.

set -e

CONFIG=
DEPENDENT_CONFIG=
IMAGE=
DEBUG=false
SUBSAMPLE=false
DRY_RUN=false

while [[ "$#" -gt 0 ]]
do case $1 in
    --config) CONFIG="$2"
    shift;;
    --dependent-config) DEPENDENT_CONFIG="$2"
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

# --debug writes no datasets and --subsample writes -subsample stores, so
# neither leaves the outputs a dependent config reads.
if [[ -n "${DEPENDENT_CONFIG}" && ( "${DEBUG}" = true || "${SUBSAMPLE}" = true ) ]]
then
    echo "Option --dependent-config cannot be combined with --debug or --subsample"
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
    -p upload_coupled_stats_script="$(< upload_coupled_stats.py)"
    -p upload_stats_script="$(< upload_stats.py)"
    -p config="$(< "${CONFIG}")"
    -p debug="${DEBUG}"
    -p subsample="${SUBSAMPLE}")

if [[ -n "${DEPENDENT_CONFIG}" ]]
then
    args+=(-p dependent_config="$(< "${DEPENDENT_CONFIG}")" -p run_dependent=true)
fi

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
