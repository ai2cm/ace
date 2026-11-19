#!/bin/bash
# Submit create_coupled_datasets.py to argo via
# create_coupled_datasets_argo_workflow.yaml, which then uploads the merged
# stats to the config's stats.beaker_dataset with upload_coupled_stats.py.
# Every module in the two entry points' transitive sibling-import closures goes
# to the workflow in one compressed code_bundle parameter: argo hands each
# resolved template to argoexec as one env string, which the kernel caps at
# MAX_ARG_STRLEN, and the modules as plain text overflow it.

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

MODULES=(create_coupled_datasets.py
    coupled_dataset_utils.py
    create_window_avg_dataset.py
    time_utils.py
    get_stats.py
    merge_stats.py
    combine_stats.py
    writer_utils.py
    fs_utils.py
    upload_coupled_stats.py
    upload_stats.py)

args=(create_coupled_datasets_argo_workflow.yaml
    -p code_bundle="$(tar -cf - "${MODULES[@]}" | gzip -9 | base64 | tr -d '\n')"
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
