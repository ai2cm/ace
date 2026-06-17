#!/bin/bash
# Launch the inline-inference forward-vs-aggregator timing benchmark on beaker.
#
# Each job runs BOTH the sequential and concurrent inference code paths
# back-to-back in one process on one GPU, with per-phase CUDA-event timers, so
# there is no co-located A/B (the confounder in the 2026-06-16 log reanalysis).
# Submit one job per node and use the repeats across jobs to clear node-to-node
# variance. Results (per-phase wall/GPU time + speedups) are written to
# /results/bench-<name>.json in each job's beaker result dataset.

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to repo root
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

CONFIG="$SCRIPT_PATH/bench-pair-era5-4deg-daily.yaml"

# Benchmark knobs (override via env): full production rollout length, 3 repeats.
REPEATS="${REPEATS:-3}"
WARMUP="${WARMUP:-1}"
MAX_WINDOWS="${MAX_WINDOWS:-0}"   # 0 = full production length
MODES="${MODES:-sequential,concurrent}"

run_benchmark() {
  local job_name="$1"
  local WORKSPACE="${2:-ai2/climate-titan}"
  local PRIORITY="${3:-urgent}"
  local CLUSTER="${4:-ai2/titan}"

  local cluster_args=()
  for c in $CLUSTER; do
    cluster_args+=(--cluster "$c")
  done

  gantry run \
    --name "$job_name" \
    --description 'Inline-inference forward-vs-aggregator timing benchmark (seq vs concurrent)' \
    --beaker-image "$(cat "$REPO_ROOT"/latest_deps_only_image.txt)" \
    --workspace "$WORKSPACE" \
    --priority "$PRIORITY" \
    "${cluster_args[@]}" \
    --gpus 1 \
    --shared-memory 400GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --allow-dirty \
    --system-python \
    --install "pip install --no-deps ." \
    -- python -I -m fme.ace.train.benchmark_inline_inference \
        "$CONFIG" \
        --repeats "$REPEATS" \
        --warmup "$WARMUP" \
        --max-windows "$MAX_WINDOWS" \
        --modes "$MODES" \
        --output-dir /results/diagnostics \
        --output-json "/results/bench-${job_name}.json"
}

# One job per node (titan B200, urgent). Two jobs to clear node-to-node variance.
run_benchmark "cinf-timing-bench-n1"
run_benchmark "cinf-timing-bench-n2"
