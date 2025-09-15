#!/bin/bash

set -e

usage() {
    cat <<EOF
Usage: $(basename "$0") DATASET_ID

Copies a Beaker dataset into the /climate-default weka directory under /climate-default/beaker-datasets/[DATASET_ID].

Arguments:
  DATASET_ID   The source dataset ID or name to copy.

Options:
  -h, --help   Show this help message and exit.

Example:
  $(basename "$0") 01K2K9VSJ17NGRZRY9CZEGREW7
EOF
}

# Show help if requested or no args
if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

SRC_DATASET=$1
REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT  # so paths are valid no matter where we are running this script

beaker dataset get $SRC_DATASET && gantry run \
    --name "copy-dataset" \
    --task-name "copy-dataset" \
    --description "Copy dataset" \
    --docker-image='python:3.10' \
    --workspace ai2/ace \
    --priority normal \
    --preemptible \
    --cluster ai2/phobos \
    --gpus 0 \
    --dataset $SRC_DATASET:/src-dataset \
    --weka climate-default:/climate-default \
    --budget ai2/climate \
    --no-python \
    --install "echo 'skipping installation step'" \
    -- bash -c "mkdir -p /climate-default/beaker-datasets/$SRC_DATASET && cp -r /src-dataset/* /climate-default/beaker-datasets/$SRC_DATASET"