#!/bin/bash
# Test-set inference for video-pmd-spatiotemporal-25km-100km-global-5ch-flat-v1:
# GLOBAL (not patch-tiled) 4-member ensemble infilling over the full held-out
# test period (2023-01-01 .. 2024-01-04), via gantry + torchrun DDP on
# ai2/titan. Reads data from weka, reads the trained checkpoint from its
# Beaker result dataset, writes the output zarr back to weka. See
# video_inference.yaml's header for why this is global instead of patched.
#
# Checkpoint dataset (current result dataset of the RESUMED run
# 01KYQG2PTD4C9PS5PYFBBS3ZF9, itself resumed from the original killed run
# 01KYGFN96W7XXJGVX8H9YWZYN3):
#   01KYQG2PTJKRGNWZXTMR6MVC8B
#
# Output:
#   /climate-default/2026-06-25-temporal-diffusion/inference/video-pmd-spatiotemporal-25km-100km-global-5ch-flat-v1/test-2023-2024-ens4.zarr
#
# Prereqs (one-time, PER WORKSPACE -- secrets are workspace-scoped):
#   pip install beaker-gantry
#   also commit + push your code: gantry runs your pushed git commit.
#
# Run:  bash configs/experiments/2026-07-24-video-pmd-spatiotemporal-25km-100km-test-inference/run.sh
set -e

JOB_NAME="video-pmd-spatiotemporal-25km-100km-global-5ch-flat-v1-test-inference"
CONFIG_FILENAME="video_inference.yaml"
WORKSPACE="ai2/climate-titan"
CLUSTER="ai2/titan"
N_GPUS=4
CHECKPOINT_DATASET="01KYQG2PTJKRGNWZXTMR6MVC8B"
WANDB_SECRET="CHLOE_WANDB_API_KEY"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_PATH="${SCRIPT_DIR#"$REPO_ROOT"/}/$CONFIG_FILENAME"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run \
    --name "$JOB_NAME" \
    --description 'Video PMD spatiotemporal test-set inference, GLOBAL (not patched), 4-member ensemble, flat/independent noise, 5 channels, 25km/100km. 4x GPU DDP on titan.' \
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
