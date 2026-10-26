#!/bin/bash
# Submit a Swin speed benchmark (bench_swin.py or bench_real.py) as a 1-GPU
# Beaker job. Usage:
#   ./run-swin-bench.sh <job_name containing "test"> <script.py> <cluster|all> <n_gpus> [args...]
# e.g. cluster "ai2/titan"; "all" allows titan, jupiter and saturn.

set -e
command -v gantry >/dev/null || export PATH="$HOME/mamba/envs/fme/bin:$PATH"

JOB_NAME="$1"
BENCH_SCRIPT="$2"
CLUSTERS="$3"
N_GPUS="$4"
shift 4
if [[ "$JOB_NAME" != *test* ]]; then
  echo "job name must contain 'test'" >&2
  exit 1
fi
SCRIPT_DIR=configs/experiments/2026-10-19-fm/swin_speed/
REPO_ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
if [[ "$CLUSTERS" == all ]]; then
  CLUSTERS="ai2/titan ai2/jupiter ai2/saturn"
fi
cd "$REPO_ROOT"

cluster_args=()
for cluster in $CLUSTERS; do
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
  --gpus "$N_GPUS" \
  --shared-memory 100GiB \
  --weka climate-default:/climate-default \
  --budget ai2/atec-climate \
  --system-python \
  --install "pip install --no-deps ." \
  --allow-dirty \
  -- torchrun --nproc_per_node "$N_GPUS" "${SCRIPT_DIR}${BENCH_SCRIPT}" "$@"
