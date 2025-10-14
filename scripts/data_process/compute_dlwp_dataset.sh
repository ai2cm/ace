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
GANTRY=false

while [[ "$#" -gt 0 ]]
do
    case $1 in
        --config) CONFIG="$2"
                  shift;;
        --argo) ARGO="$2"
                shift;;
        --gantry) GANTRY=true
                  shift;;
        --overwrite) OVERWRITE=true
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

run_directory=$(yq -r '.data_output_directory' ${CONFIG})
output_directory="${run_directory}_dlwp"

if [[ "$ARGO" == "true" ]]
then
    echo "Argo workflow not supported for compute_dlwp_dataset"
elif [[ "$GANTRY" == "true" ]]
then
    JOB_NAME="compute-dlwp-dataset"
    BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
    REPO_ROOT=$(git rev-parse --show-toplevel)
    cd $REPO_ROOT
    gantry run --allow-dirty --no-python \
        --name $JOB_NAME \
        --task-name $JOB_NAME \
        --not-preemptible \
        --description 'Run DLWP dataset computation' \
        --priority normal \
        --beaker-image annad/dlwp-ace-datapipe-v2025.09.0 \
        --workspace ai2/ace \
        --cluster ai2/ceres-cirrascale \
        --cluster ai2/saturn-cirrascale \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --gpus 0 \
        --shared-memory 400GiB \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        -- bash -c "gcloud auth activate-service-account --key-file=/tmp/google_application_credentials.json && python3 ./scripts/data_process/compute_dlwp_dataset.py --config=\"./scripts/data_process/${CONFIG}\" --run-directory=\"${run_directory}\" --output-store=\"${output_directory}\" $(if [[ \"$OVERWRITE\" == \"true\" ]]; then echo \"--overwrite\"; fi) && echo '=== DLWP ACE DATASET COMPUTATION COMPLETED SUCCESSFULLY ===' && echo \"Output written to: ${output_directory}\" && echo \"Config used: ${CONFIG}\" && echo \"Timestamp: \$(date)\""
else
    if [[ "$OVERWRITE" == "true" ]]; then
    python3 compute_dlwp_dataset.py --config="${CONFIG}" \
            --run-directory="${run_directory}" \
            --output-store="${output_directory}" \
            --overwrite
    else
        python3 compute_dlwp_dataset.py --config="${CONFIG}" \
            --run-directory="${run_directory}" \
            --output-store="${output_directory}"
    fi
fi
