#!/bin/bash


while [[ "$#" -gt 0 ]]
do case $1 in
    --root) ROOT="$2"
    shift;;
    --output) OUTPUT="$2"
    shift;;
    --ic-max) IC_MAX="$2"
    shift;;
    --script-flags) SCRIPT_FLAGS="$2"
    shift;;
    *) echo "Unknown parameter passed: $1"
    exit 1;;
esac
shift
done

if [[ -z "${ROOT}" ]]
then
    echo "Option --root missing"
    exit 1;
elif [[ -z "${OUTPUT}" ]]
then
    echo "Option --output missing"
    exit 1;
elif [[ -z "${IC_MAX}" ]]
then
    IC_MAX=1
fi

argo submit compute_dataset_fv3gfs_argo_workflow.yaml \
    -p python_script="$(< compute_dataset_fv3gfs.py)" \
    -p root="${ROOT}" -p output="${OUTPUT}" -p ic_max=${IC_MAX} \
    -p script_flags="${SCRIPT_FLAGS}"
