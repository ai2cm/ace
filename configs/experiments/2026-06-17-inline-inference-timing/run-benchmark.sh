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

# Derive paths from the script's own location (works regardless of cwd), so the
# config path passed to the job is correct repo-relative when the cloned job
# runs from the repo root.
SCRIPT_DIR_ABS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR_ABS" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
SCRIPT_PATH="${SCRIPT_DIR_ABS#"$REPO_ROOT"/}"  # repo-relative dir of this script

CONFIG="$SCRIPT_PATH/bench-pair-era5-4deg-daily.yaml"

# Benchmark knobs (override via env): full production rollout length, 3 repeats.
REPEATS="${REPEATS:-3}"
WARMUP="${WARMUP:-1}"
MAX_WINDOWS="${MAX_WINDOWS:-0}"   # 0 = full production length
MODES="${MODES:-sequential,concurrent}"
SUFFIX="${SUFFIX:-}"   # appended to job names to avoid beaker name collisions

run_benchmark() {
  local job_name="$1"
  local WORKSPACE="${2:-ai2/climate-titan}"
  local PRIORITY="${3:-urgent}"
  local CLUSTER="${4:-ai2/titan}"

  # NOTE on node isolation: gantry's --weka auto-adds a `cluster` constraint,
  # and Beaker rejects `cluster` + `hostname` together, so we cannot pin a host
  # while mounting weka data. That is fine here: each job runs BOTH code paths
  # back-to-back in one process on one dedicated GPU, so the
  # sequential-vs-concurrent A/B is never co-located (the confound the 2026-06-16
  # log reanalysis hit), and the headline forward/aggregator numbers are
  # CUDA-event GPU times, robust to a sibling job on another GPU of the host. To
  # truly place the two repeat-jobs on distinct hosts, submit them and check
  # `beaker experiment get ... node`, stopping/resubmitting until they differ.
  local placement_args=()
  for c in $CLUSTER; do
    placement_args+=(--cluster "$c")
  done

  gantry run \
    --name "$job_name" \
    --description 'Inline-inference forward-vs-aggregator timing benchmark (seq vs concurrent)' \
    --beaker-image "$(cat "$REPO_ROOT"/latest_deps_only_image.txt)" \
    --workspace "$WORKSPACE" \
    --priority "$PRIORITY" \
    "${placement_args[@]}" \
    --gpus 1 \
    --shared-memory 400GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --allow-dirty \
    --system-python \
    --install "pip install --no-deps ." \
    --yes --no-logs --timeout 0 \
    -- python -I -m fme.ace.train.benchmark_inline_inference \
        "$CONFIG" \
        --repeats "$REPEATS" \
        --warmup "$WARMUP" \
        --max-windows "$MAX_WINDOWS" \
        --modes "$MODES" \
        --output-dir /results/diagnostics \
        --output-json "/results/bench-${job_name}.json"
}

# Two jobs (titan B200, urgent) for node-to-node variance. Each job runs both
# code paths internally, so there is never a co-located seq-vs-concurrent A/B.
run_benchmark "cinf-timing-bench-r1${SUFFIX}"
run_benchmark "cinf-timing-bench-r2${SUFFIX}"
