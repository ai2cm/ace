#!/bin/bash
# Submit the Swin training-step microbenchmark (bench_swin.py) as a 1-GPU
# Beaker job. Usage: ./run-swin-bench.sh <job_name containing "test"> [bench args...]

set -e
command -v gantry >/dev/null || export PATH="$HOME/mamba/envs/fme/bin:$PATH"

JOB_NAME="$1"
shift
if [[ "$JOB_NAME" != *test* ]]; then
  echo "job name must contain 'test'" >&2
  exit 1
fi
SCRIPT_DIR=configs/experiments/2026-10-19-fm/swin_speed/
REPO_ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
BEAKER_CLUSTER=${BEAKER_CLUSTER:-"ai2/titan ai2/jupiter ai2/saturn"}
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
