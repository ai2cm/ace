#!/bin/bash
# Test-set inference for video-pmd-spatiotemporal-25km-100km-global-5ch-ou-v2,
# region: NORTHERN polar cap, lat 44..88, full longitude -- see video_inference.yaml's header for the full
# tiling/checkpoint-caveat rationale (crashed at epoch 41/200, run as-is
# per explicit instruction, NOT resumed).
#
# Checkpoint dataset: 01KZ9JHSJNRWXSMX8H1EGZSTYR
#
# Output:
#   /climate-default/2026-06-25-temporal-diffusion/inference/video-pmd-spatiotemporal-25km-100km-global-5ch-ou-v2/test-2023-2024-ens4-region-lat44to88-lon0to360.zarr
#
# Prereqs (one-time, PER WORKSPACE -- secrets are workspace-scoped):
#   pip install beaker-gantry
#   also commit + push your code: gantry runs your pushed git commit.
#
# Run:  bash configs/experiments/2026-08-10-video-pmd-spatiotemporal-25km-100km-ou-v2-test-inference-latNcap/run.sh
set -e

JOB_NAME="video-pmd-spatiotemporal-25km-100km-global-5ch-ou-v2-test-inference-latNcap"
CONFIG_FILENAME="video_inference.yaml"
WORKSPACE="ai2/climate-titan"
CLUSTER="ai2/titan"
N_GPUS=4
CHECKPOINT_DATASET="01KZ9JHSJNRWXSMX8H1EGZSTYR"
WANDB_SECRET="CHLOE_WANDB_API_KEY"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_PATH="${SCRIPT_DIR#"$REPO_ROOT"/}/$CONFIG_FILENAME"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run --allow-dirty \
    --name "$JOB_NAME" \
    --description 'Video PMD spatiotemporal v2 test-set inference, NORTHERN polar cap, lat 44..88, full longitude, 4-member ensemble, per-channel Ornstein-Uhlenbeck noise kernel, 5 channels, 25km/100km, LR-endpoints-in/HR-full-out. Checkpoint is epoch 41/200 (crashed, not resumed -- see yaml header). 4x GPU DDP on titan.' \
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
