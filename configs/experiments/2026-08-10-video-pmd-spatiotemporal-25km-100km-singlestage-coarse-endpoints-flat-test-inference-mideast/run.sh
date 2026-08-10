#!/bin/bash
# Test-set inference for
# video-pmd-spatiotemporal-25km-100km-global-5ch-singlestage-coarse-endpoints-flat,
# region: mid-latitude EAST quarter, lat -44..44, lon 180..360 -- see video_inference.yaml's header for the full
# architecture/tiling/checkpoint-caveat rationale. NOT YET LAUNCHED -- this
# is prepared ahead of confirming training has produced a good-enough
# checkpoint.
#
# Checkpoint dataset: 01KZEN3KJVNMCQMPAEB8KXWRA3
#
# Output:
#   /climate-default/2026-06-25-temporal-diffusion/inference/video-pmd-spatiotemporal-25km-100km-global-5ch-singlestage-coarse-endpoints-flat/test-2023-2024-ens4-region-lat-44to44-lon180to360.zarr
#
# Prereqs (one-time, PER WORKSPACE -- secrets are workspace-scoped):
#   pip install beaker-gantry
#   also commit + push your code: gantry runs your pushed git commit.
#
# Run:  bash configs/experiments/2026-08-10-video-pmd-spatiotemporal-25km-100km-singlestage-coarse-endpoints-flat-test-inference-mideast/run.sh
set -e

JOB_NAME="video-pmd-spatiotemporal-25km-100km-global-5ch-singlestage-coarse-endpoints-flat-test-inference-mideast"
CONFIG_FILENAME="video_inference.yaml"
WORKSPACE="ai2/climate-titan"
CLUSTER="ai2/titan"
N_GPUS=2
CHECKPOINT_DATASET="01KZEN3KJVNMCQMPAEB8KXWRA3"
WANDB_SECRET="CHLOE_WANDB_API_KEY"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_PATH="${SCRIPT_DIR#"$REPO_ROOT"/}/$CONFIG_FILENAME"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run --allow-dirty \
    --name "$JOB_NAME" \
    --description 'Video PMD spatiotemporal SINGLE-STAGE (true LR-endpoints-in/HR-full-out) test-set inference, mid-latitude EAST quarter, lat -44..44, lon 180..360, 4-member ensemble, flat/independent noise, 5 channels, 25km/100km. Checkpoint is preliminary/undertrained -- see yaml header. 2x GPU DDP on titan.' \
    --workspace "$WORKSPACE" \
    --priority urgent \
    --cluster "$CLUSTER" \
    --beaker-image "$DEPS_ONLY_IMAGE" \
    --gpus "$N_GPUS" \
    --shared-memory 64GiB \
    --budget ai2/atec-climate \
    --weka climate-default:/climate-default \
    --dataset "${CHECKPOINT_DATASET}:/checkpoint" \
    --env-secret WANDB_API_KEY="$WANDB_SECRET" \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.downscaling.video_inference "$CONFIG_PATH"
