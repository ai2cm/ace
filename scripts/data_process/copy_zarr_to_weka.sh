#!/bin/bash
set -e

make_gsutil_cmds() {
    local output_directory="$1"   # first arg: output directory
    local weka_run_name="$2"      # second arg: run name for weka
    shift 2                       # rest of args: names
    local names=("$@")

    local cmd=""
    for name in "${names[@]}"; do
        local weka_root="/climate-default${weka_run_name}"
        local part="mkdir -p ${weka_root} && gsutil -m -o Credentials:gs_service_key_file=/tmp/google_application_credentials.json cp -r ${output_directory}/${name}.zarr ${weka_root}"
        if [[ -z "$cmd" ]]; then
            cmd="$part"
        else
            cmd="$cmd && $part"
        fi
    done

    echo "$cmd"
}

extract_path_suffix() {
    local input="$1"

    # Remove the prefix "gs://vcm-ml-intermediate" (with or without trailing slash)
    local suffix="${input#gs://vcm-ml-intermediate/}"
    suffix="${suffix#gs://vcm-ml-intermediate}"

    suffix="/${suffix}/"  # add leading and ending slashes if exists
    if [[ -z "$suffix" ]]; then
        suffix="/"  # return /
    fi

    # Return the suffix ("/" if input was only the prefix)
    echo "$suffix"
}

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
output_directory=$(yq -r '.data_output_directory' ${CONFIG})

JOB_NAME="${CONFIG#configs/}"
JOB_NAME="${JOB_NAME%.yaml}"
JOB_NAME=copying-$JOB_NAME

REPO_ROOT=$(git rev-parse --show-toplevel)

weka_run_name=$(extract_path_suffix $output_directory)
cmds=$(make_gsutil_cmds $output_directory $weka_run_name "${names[@]}")

cd $REPO_ROOT && gantry run \
    --name $JOB_NAME \
    --description "Copy $names to WEKA" \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority normal \
    --cluster ai2/phobos \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --gpus 0 \
    --shared-memory 40GiB \
    --weka climate-default:/climate-default \
    --budget ai2/climate \
    --install "pip install --no-deps ." \
    --allow-dirty \
    -- bash -c "$cmds"
