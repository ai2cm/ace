#!/bin/bash
# Usage: bash run.sh <cluster> [<cluster> ...]   e.g. bash run.sh ai2/phobos ai2/jupiter
# Raw write throughput to the weka mount and to the node's local disk, one job per
# cluster, no GPU, no W&B. Results are in the job log.

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
REPO_ROOT=$(git rev-parse --show-toplevel)
COMMIT=$(git rev-parse --short HEAD)
WEKA_ROOT=/climate-default/home/brianhenn/ace/scratch/2026-09-29-weka-raw-write-test

cd "$REPO_ROOT"

run_cluster() {
    local cluster="$1"
    local short="${cluster#ai2/}"
    gantry run \
        --name "weka-raw-write-test-${short}-${COMMIT}" \
        --description 'Raw filesystem write throughput, weka vs local disk' \
        --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
        --workspace ai2/ace \
        --priority normal \
        --cluster "$cluster" \
        --gpus 0 \
        --weka climate-default:/climate-default \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- bash -c "
            python3 $SCRIPT_PATH/write_test.py $WEKA_ROOT --label weka &&
            python3 $SCRIPT_PATH/write_test.py $WEKA_ROOT --label weka-fsync --fsync --streams 1 4 &&
            python3 $SCRIPT_PATH/write_test.py /tmp/weka-raw-write-test --label local --gb-per-config 3
        "
}

for cluster in "$@"; do
    run_cluster "$cluster"
done
