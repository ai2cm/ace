#!/bin/bash
# Submit the Swin training-step microbenchmark (bench_swin.py) as a 1-GPU
# Beaker job. Usage: ./run-swin-bench.sh <job_name containing "test"> [bench args...]

set -e

JOB_NAME="$1"
shift
if [[ "$JOB_NAME" != *test* ]]; then
  echo "job name must contain 'test'" >&2
  exit 1
fi
SCRIPT_DIR=$(git rev-parse --show-prefix)
REPO_ROOT=$(git rev-parse --show-toplevel)
BEAKER_CLUSTER=${BEAKER_CLUSTER:-"ai2/jupiter"}
cd "$REPO_ROOT"

cluster_args=()
for cluster in $BEAKER_CLUSTER; do
  cluster_args+=(--cluster "$cluster")
done

gantry run \
  --name "$JOB_NAME" \
  --task-name "$JOB_NAME" \
  --description 'Swin speed microbenchmark' \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace \
  --priority "${BEAKER_PRIORITY:-normal}" \
  "${cluster_args[@]}" \
  --gpus 1 \
  --shared-memory 50GiB \
  --budget ai2/atec-climate \
  --system-python \
  --install "pip install --no-deps ." \
  --allow-dirty \
  -- python "${SCRIPT_DIR}bench_swin.py" "$@"
