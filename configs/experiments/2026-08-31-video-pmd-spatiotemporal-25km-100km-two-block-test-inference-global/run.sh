#!/bin/bash
# Test-set inference for
# video-pmd-spatiotemporal-25km-100km-global-5ch-two-block-coarse-endpoints-flat,
# GLOBAL DOMAIN IN ONE JOB via divide_generation -- see video_inference.yaml's
# header for the full rationale/caveats (checkpoint is from a manually
# stopped run at epoch ~54/200).
#
# Checkpoint dataset: 01M100MWQDFSZHWAQW1ZTJZFJ4
#
# Output:
#   /climate-default/2026-06-25-temporal-diffusion/inference/video-pmd-spatiotemporal-25km-100km-global-5ch-two-block-coarse-endpoints-flat/test-2023-2024-ens4-global.zarr
#
# Prereqs (one-time, PER WORKSPACE -- secrets are workspace-scoped):
#   pip install beaker-gantry
#   also commit + push your code: gantry runs your pushed git commit.
#
# Run:  bash configs/experiments/2026-08-31-video-pmd-spatiotemporal-25km-100km-two-block-test-inference-global/run.sh
set -e

JOB_NAME="video-pmd-spatiotemporal-25km-100km-global-5ch-two-block-coarse-endpoints-flat-test-inference-global"
CONFIG_FILENAME="video_inference.yaml"
WORKSPACE="ai2/ace"
CLUSTER="ai2/neptune"  # l40s -- jupiter (h100) has been starved at ~0-2/984 free
# for 2+ hours; this is a multi-day job (~19min per batch*ensemble-member
# unit measured on a single L40s smoke test x 368 batches x 4 ensemble
# members / 4 GPUs =~ 4.9 GPU-days/rank), so starting now on available
# hardware beats queuing indefinitely for faster hardware.
N_GPUS=4
CHECKPOINT_DATASET="01M100MWQDFSZHWAQW1ZTJZFJ4"
# No WANDB_API_KEY secret in ai2/ace -- fine, this config has log_to_wandb: false.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_PATH="${SCRIPT_DIR#"$REPO_ROOT"/}/$CONFIG_FILENAME"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run --allow-dirty \
    --name "$JOB_NAME" \
    --description 'Video PMD spatiotemporal TWO-BLOCK (pinned coarse-temporal r + unpinned fine-detail d, fixed kernels) test-set inference, GLOBAL DOMAIN in one job via patch-tiled divide_generation, 4-member ensemble, coarse-endpoints-only input, 5 channels, 25km/100km. Checkpoint from a manually stopped run at epoch ~54/200 -- see yaml header. Multi-day job (~4.9 GPU-days/rank measured via smoke test); 4x GPU DDP on neptune (l40s, available now vs. jupiter starved).' \
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
