#!/bin/bash

set -e

while [[ "$#" -gt 0 ]]
do case $1 in
    --start-datetime) START_DATETIME="$2"
    shift;;
    --end-datetime) END_DATETIME="$2"
    shift;;
    --destination) DESTINATION="$2"
    shift;;
    --n-workers) N_WORKERS="$2"
    shift;;
    *) echo "Unknown parameter passed: $1"
    exit 1;;
esac
shift
done

if [[ -z "${START_DATETIME}" ]]
then
    echo "Option --start-datetime missing"
    exit 1;
fi

if [[ -z "${END_DATETIME}" ]]
then
    echo "Option --end-datetime missing"
    exit 1;
fi

if [[ -z "${DESTINATION}" ]]
then
    echo "Option --destination missing"
    exit 1;
fi

if [[ -z "${N_WORKERS}" ]]
then
    echo "Option --n-workers missing"
    exit 1;
fi

# Capture the output of the argo submit command
output=$(argo submit argo_workflow.yaml \
    -p python_script="$(< process.py)" \
    -p start_datetime="${START_DATETIME}" \
    -p end_datetime="${END_DATETIME}" \
    -p destination="${DESTINATION}" \
    -p n_workers="${N_WORKERS}")

# Extract the job name from the output
job_name=$(echo "$output" | grep 'Name:' | awk '{print $2}')

# Print the job name
echo "Argo job submitted: $job_name"
