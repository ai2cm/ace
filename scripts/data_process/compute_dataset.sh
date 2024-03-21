#!/bin/bash


while [[ "$#" -gt 0 ]]
do case $1 in
    --config) CONFIG="$2"
    shift;;
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
output_directory=$(yq -r '.output_directory' ${CONFIG})
ic_max=$(yq -r '.runs | length' ${CONFIG})

argo submit compute_dataset_argo_workflow.yaml \
    -p python_script="$(< compute_dataset.py)" \
    -p config="$(< ${CONFIG})" \
    -p names="${names[*]}" \
    -p run_directories="${run_directories[*]}" \
    -p output_directory="${output_directory}" \
    -p ic_max=${ic_max}
