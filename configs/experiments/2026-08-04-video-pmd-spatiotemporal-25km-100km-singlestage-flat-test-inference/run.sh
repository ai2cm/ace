#!/bin/bash
# Test-set inference for global-25km-100km-spatiotemporal-singlestage-5ch-flat-v1,
# region 1 of 4 (lat -44..44, lon 0..180) -- see video_inference.yaml's
# header for the tiling/checkpoint rationale.
#
# Checkpoint dataset: 01KYXJ16PKMQJN965WWTSFAQ09 (latest.ckpt)
#
# Output:
#   /climate-default/2026-06-25-temporal-diffusion/inference/global-25km-100km-spatiotemporal-singlestage-5ch-flat-v1/test-2023-2024-ens4-region-lat-44to44-lon0to180.zarr
#
# Prereqs (one-time, PER WORKSPACE -- secrets are workspace-scoped):
#   pip install beaker-gantry
#   also commit + push your code: gantry runs your pushed git commit.
#
# Run:  bash configs/experiments/2026-08-04-video-pmd-spatiotemporal-25km-100km-singlestage-flat-test-inference/run.sh
set -e

JOB_NAME="global-25km-100km-spatiotemporal-singlestage-5ch-flat-v1-test-inference"
CONFIG_FILENAME="video_inference.yaml"
WORKSPACE="ai2/climate-titan"
CLUSTER="ai2/titan"
N_GPUS=4
CHECKPOINT_DATASET="01KYXJ16PKMQJN965WWTSFAQ09"
WANDB_SECRET="CHLOE_WANDB_API_KEY"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_PATH="${SCRIPT_DIR#"$REPO_ROOT"/}/$CONFIG_FILENAME"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run --allow-dirty \
    --name "$JOB_NAME" \
    --description 'Video PMD spatiotemporal SINGLE-STAGE test-set inference, region 1/4 (lat -44..44, lon 0..180), 4-member ensemble, flat/independent noise, 5 channels, 25km/100km. 4x GPU DDP on titan.' \
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
