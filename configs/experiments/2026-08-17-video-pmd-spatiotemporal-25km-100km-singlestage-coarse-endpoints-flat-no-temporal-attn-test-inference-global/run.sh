#!/bin/bash
# Test-set inference for
# video-pmd-spatiotemporal-25km-100km-global-5ch-singlestage-coarse-endpoints-flat-no-temporal-attn,
# GLOBAL DOMAIN IN ONE JOB via divide_generation -- see video_inference.yaml's
# header for the full rationale/caveats.
#
# Checkpoint dataset: 01M00Z9062AYGEBHTFQJJVTWTZ
#
# Output:
#   /climate-default/2026-06-25-temporal-diffusion/inference/video-pmd-spatiotemporal-25km-100km-global-5ch-singlestage-coarse-endpoints-flat-no-temporal-attn/test-2023-2024-ens4-global.zarr
#
# Prereqs (one-time, PER WORKSPACE -- secrets are workspace-scoped):
#   pip install beaker-gantry
#   also commit + push your code: gantry runs your pushed git commit.
#
# Run:  bash configs/experiments/2026-08-17-video-pmd-spatiotemporal-25km-100km-singlestage-coarse-endpoints-flat-no-temporal-attn-test-inference-global/run.sh
set -e

JOB_NAME="video-pmd-spatiotemporal-25km-100km-global-5ch-singlestage-coarse-endpoints-flat-no-temporal-attn-test-inference-global"
CONFIG_FILENAME="video_inference.yaml"
WORKSPACE="ai2/climate-titan"
CLUSTER="ai2/ceres"  # h100
N_GPUS=4
CHECKPOINT_DATASET="01M00Z9062AYGEBHTFQJJVTWTZ"
WANDB_SECRET="CHLOE_WANDB_API_KEY"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_PATH="${SCRIPT_DIR#"$REPO_ROOT"/}/$CONFIG_FILENAME"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run --allow-dirty \
    --name "$JOB_NAME" \
    --description 'Video PMD spatiotemporal SINGLE-STAGE coarse-endpoints, NO-TEMPORAL-ATTENTION ablation, test-set inference, GLOBAL DOMAIN in one job via patch-tiled divide_generation, 4-member ensemble, flat/independent noise, 5 channels, 25km/100km. Checkpoint fully trained (31/31 epochs, exitCode 0). 4x GPU DDP on jupiter (h100).' \
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
