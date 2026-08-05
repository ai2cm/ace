#!/bin/bash
# Test-set inference for video-pmd-spatiotemporal-25km-100km-global-5ch-ou-v1,
# region 3 of 4 (northern polar cap, lat 44..88, lon 0..360) -- see
# video_inference.yaml's header for the tiling rationale. Same
# checkpoint/settings as region 1
# (../2026-07-27-video-pmd-spatiotemporal-25km-100km-ou-test-inference/),
# only lat_extent/lon_extent and output_path differ.
#
# Checkpoint dataset (same as region 1): 01KYQG329F629C70X0QH2WKZPZ
#
# Output:
#   /climate-default/2026-06-25-temporal-diffusion/inference/video-pmd-spatiotemporal-25km-100km-global-5ch-ou-v1/test-2023-2024-ens4-region-lat44to88-lon0to360.zarr
#
# Prereqs (one-time, PER WORKSPACE -- secrets are workspace-scoped):
#   pip install beaker-gantry
#   also commit + push your code: gantry runs your pushed git commit.
#
# Run:  bash configs/experiments/2026-08-03-video-pmd-spatiotemporal-25km-100km-ou-test-inference-latNcap/run.sh
set -e

JOB_NAME="video-pmd-spatiotemporal-25km-100km-global-5ch-ou-v1-test-inference-latNcap"
CONFIG_FILENAME="video_inference.yaml"
WORKSPACE="ai2/climate-titan"
CLUSTER="ai2/titan"
N_GPUS=4
CHECKPOINT_DATASET="01KYQG329F629C70X0QH2WKZPZ"
WANDB_SECRET="CHLOE_WANDB_API_KEY"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_PATH="${SCRIPT_DIR#"$REPO_ROOT"/}/$CONFIG_FILENAME"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run --allow-dirty \
    --name "$JOB_NAME" \
    --description 'Video PMD spatiotemporal test-set inference, region 3/4 (northern polar cap, lat 44..88, lon 0..360), 4-member ensemble, OU per-channel noise, 5 channels, 25km/100km. 4x GPU DDP on titan.' \
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
