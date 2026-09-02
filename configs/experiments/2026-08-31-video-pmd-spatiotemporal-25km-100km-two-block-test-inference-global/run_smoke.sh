#!/bin/bash
# SMOKE test isolating the NCCL ALLREDUCE hang seen in the full 4-GPU DDP
# global-domain run (run.sh) -- see video_inference_smoke.yaml's header.
# Single GPU (no DDP, so no cross-rank collective can block), max_batches=1,
# n_ensemble=1. If this also hangs, it's a plain slowness/infinite-loop bug
# in the new divide_generation path, unrelated to distributed sync; if it
# completes quickly, the hang is specific to multi-rank DDP.
#
# Run:  bash configs/experiments/2026-08-31-video-pmd-spatiotemporal-25km-100km-two-block-test-inference-global/run_smoke.sh
set -e

JOB_NAME="video-pmd-spatiotemporal-25km-100km-global-5ch-two-block-coarse-endpoints-flat-test-inference-global-smoke"
CONFIG_FILENAME="video_inference_smoke.yaml"
WORKSPACE="ai2/ace"
CLUSTER="ai2/jupiter"  # h100
N_GPUS=1
CHECKPOINT_DATASET="01M100MWQDFSZHWAQW1ZTJZFJ4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_PATH="${SCRIPT_DIR#"$REPO_ROOT"/}/$CONFIG_FILENAME"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run --allow-dirty \
    --name "$JOB_NAME" \
    --description 'SMOKE test (1 GPU, 1 batch, 1 ensemble member) for the two-block global divide_generation inference config, diagnosing an NCCL ALLREDUCE hang seen on the 4-GPU DDP run.' \
    --workspace "$WORKSPACE" \
    --priority urgent \
    --cluster "$CLUSTER" \
    --beaker-image "$DEPS_ONLY_IMAGE" \
    --gpus "$N_GPUS" \
    --shared-memory 64GiB \
    --budget ai2/atec-climate \
    --weka climate-default:/climate-default \
    --dataset "${CHECKPOINT_DATASET}:/checkpoint" \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.downscaling.video_inference "$CONFIG_PATH"
