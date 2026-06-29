#!/bin/bash
# CPU-only Beaker smoke run for the video PMD trainer, via gantry.
#
# Reads data from weka (climate-default), CPU-only (--gpus 0, FME_FORCE_CPU=1),
# single process. Mirrors the weka pattern used by other ACE experiment runs
# (e.g. configs/experiments/2025-11-15-ace2s-x-shield/run-ace2s-train.sh).
#
# Data on weka:
#   /climate-default/2026-06-25-temporal-diffusion/2025-07-25-X-SHiELD-AMIP-FME-3h.zarr
#
# Prereqs (one-time):
#   pip install beaker-gantry            # if gantry isn't installed
#   beaker secret write --workspace ai2/chloeh CHLOE_WANDB_API_KEY <your-wandb-key>
#
# Run:  bash configs/experiments/2026-06-29-video-pmd-smoke/run.sh
set -e

JOB_NAME="video-pmd-regional-30-60-1degree-24to3-cpu-smoke"
CONFIG_FILENAME="video_train_smoke.yaml"
WORKSPACE="ai2/chloeh"
CLUSTER="ai2/phobos"
WANDB_SECRET="CHLOE_WANDB_API_KEY"   # beaker secret name holding your W&B API key

# Resolve the config path relative to the repo root from the script's OWN
# location, so this works whether invoked from the repo root or the script dir.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_PATH="${SCRIPT_DIR#"$REPO_ROOT"/}/$CONFIG_FILENAME"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run \
    --name "$JOB_NAME" \
    --description 'CPU smoke: endpoint-conditioned video interpolation (PMD), regional 30x60 1deg, 24h->3h' \
    --workspace "$WORKSPACE" \
    --priority normal \
    --preemptible \
    --cluster "$CLUSTER" \
    --beaker-image "$DEPS_ONLY_IMAGE" \
    --gpus 0 \
    --shared-memory 10GiB \
    --budget ai2/atec-climate \
    --weka climate-default:/climate-default \
    --env FME_FORCE_CPU=1 \
    --env-secret WANDB_API_KEY="$WANDB_SECRET" \
    --system-python \
    --install "pip install --no-deps ." \
    -- python -m fme.downscaling.video_train "$CONFIG_PATH"
