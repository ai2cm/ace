#!/bin/bash
# Launches the PR #1519 CUDA-timer benchmark A/B (see benchmark_ab.sh) from the
# current HEAD, which must be the branch tip (base + PR merge + benchmark).

set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_PATH=$(git rev-parse --show-prefix)
JOB_NAME="pr1519-disco-cuda-benchmark-$(git rev-parse --short HEAD)"

cd $REPO_ROOT

gantry run \
  --name "$JOB_NAME" \
  --task-name "$JOB_NAME" \
  --description "PR #1519 DISCO CUDA-timer benchmark A/B, base vs PR on one GPU" \
  --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
  --workspace ai2/ace \
  --priority normal \
  --cluster ai2/jupiter \
  --gpus 1 \
  --shared-memory 50GiB \
  --budget ai2/atec-climate \
  --system-python \
  --install "pip install --no-deps ." \
  -- bash -c "cp $SCRIPT_PATH/benchmark_ab.sh /tmp/benchmark_ab.sh && bash /tmp/benchmark_ab.sh"
