#!/bin/bash

# Function to check if directory is in Python path and add it if not
function add_to_pythonpath() {
    local dir_to_add="$1"
    if [[ ":$PYTHONPATH:" != *":$dir_to_add:"* ]]; then
        export PYTHONPATH="$dir_to_add:$PYTHONPATH"
    fi
}

# Add your own full-model directory to Python path if not already included
add_to_pythonpath "~/full-model"

ARGO=false

while [[ "$#" -gt 0 ]]
do
    case $1 in
        --config) CONFIG="$2"
                  shift;;
        --argo) ARGO="$2"
                shift;;
        *) echo "Unknown parameter passed: $1"
           exit 1;;
    esac
    shift
done

if [[ -z "${CONFIG}" ]]
then
    echo "Option --config missing"
    exit 1
fi

run_directory=$(yq -r '.runs.run_directory' ${CONFIG})
output_directory=$(yq -r '.data_output_directory' ${CONFIG})

if [[ "$ARGO" == "true" ]]
then
    output=$(argo submit full-model/scripts/data_process/compute_hpx_dataset_argo_workflow.yaml \
        -p python_script="$(< full-model/scripts/data_process/compute_hpx_dataset.py)" \
        -p config="$(< ${CONFIG})" \
        -p run_directory="${run_directory}" \
        -p output_directory="${output_directory}")

    job_name=$(echo "$output" | grep 'Name:' | awk '{print $2}')
    echo "Argo job submitted: $job_name"
else
    python3 full-model/scripts/data_process/compute_hpx_dataset.py --config="${CONFIG}" \
        --run-directory="${run_directory}" \
        --output-store="${output_directory}"
fi