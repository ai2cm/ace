#!/bin/bash
# Cheap validation smoke test: does nonzero EDM churn (S_churn=20) restore
# ensemble spread for the brownian-bridge noise scheme? Single 3-day window
# (2023-01-01..2023-01-04), 32-member ensemble, 4x B200 DDP -- should take
# ~O(10 min), not the ~16.5h a full-year run would.
#
# Checkpoint dataset (same as bb-pcn's own inference run):
#   01KWDBKFBCWGKCAJD5B49H4TND
#
# Output:
#   /climate-default/2026-06-25-temporal-diffusion/inference/video-pmd-bb-pcn-churn20-global-1degree-24to3-v1/smoketest-jan2023-ens32.zarr
#
# Run:  bash configs/experiments/2026-07-20-video-pmd-bb-pcn-churn20-smoketest/run.sh
set -e

JOB_NAME="video-pmd-bb-pcn-churn20-smoketest"
CONFIG_FILENAME="video_inference.yaml"
WORKSPACE="ai2/climate-titan"
CLUSTER="ai2/titan"
N_GPUS=4
CHECKPOINT_DATASET="01KWDBKFBCWGKCAJD5B49H4TND"
WANDB_SECRET="CHLOE_WANDB_API_KEY"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_PATH="${SCRIPT_DIR#"$REPO_ROOT"/}/$CONFIG_FILENAME"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run \
    --name "$JOB_NAME" \
    --description 'Churn=20 smoke test for brownian-bridge video PMD noise (single 3-day window, 32-member ensemble)' \
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
