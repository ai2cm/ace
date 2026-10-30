#!/bin/bash

set -e

JOB_NAME="2026-10-30-samudrace-paper-ckpt-piControl-create-ic-0320"
CONFIG_FILENAME="create-ic-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH="${SCRIPT_PATH}${CONFIG_FILENAME}"
REPO_ROOT=$(git rev-parse --show-toplevel)

cd "$REPO_ROOT"  # so config path is valid no matter where we are running this script

# create_coupled_ic.py imports sibling modules and resolves its coupled dataset
# config path relative to scripts/data_process, so it must be run from there.
gantry run \
    --name $JOB_NAME \
    --task-name $JOB_NAME \
    --description "Create 0320-01-01 coupled CM4 piControl initial condition" \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority normal \
    --not-preemptible \
    --cluster ai2/ceres-cirrascale \
    --cluster ai2/saturn-cirrascale \
    --weka climate-default:/climate-default \
    --gpus 0 \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- bash -c "cd scripts/data_process && python create_coupled_ic.py --config ../../${CONFIG_PATH}"
