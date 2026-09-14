#!/bin/bash
# Test-set inference for
# video-pmd-spatiotemporal-25km-100km-global-5ch-two-block-hiro-patch,
# GLOBAL DOMAIN IN ONE JOB via divide_generation -- see video_inference.yaml's
# header for the full rationale/caveats (checkpoint stopped at epoch ~55-60,
# matching the 44x72 comparison checkpoint's own stop depth; coarse_patch_extent
# 16x16 tiles the domain into ~250 patches instead of 20).
#
# Checkpoint dataset: 01M26N8CBHE5DB0KG08WPR5EWV
#
# Output:
#   /climate-default/2026-06-25-temporal-diffusion/inference/video-pmd-spatiotemporal-25km-100km-global-5ch-two-block-hiro-patch/test-2023-2024-ens4-global.zarr
#
# Cluster: ai2/titan (B200), not ai2/neptune (L40s) like the 44x72 config --
# that config picked neptune because jupiter (H100) was starved at the time;
# titan currently has real headroom (~19/96 free) and is the fastest hardware
# available, which matters here since divide_generation over ~12.5x more
# (smaller) patches is expected to take longer than the 44x72 run's own
# ~4.9 GPU-day estimate. Re-check `beaker cluster get ai2/titan` if this sits
# queued -- fall back to ai2/jupiter (H100) if titan is unexpectedly busy.
#
# Prereqs (one-time, PER WORKSPACE -- secrets are workspace-scoped):
#   pip install beaker-gantry
#   also commit + push your code: gantry runs your pushed git commit.
#
# Run:  bash configs/experiments/2026-09-15-video-pmd-spatiotemporal-25km-100km-two-block-hiro-patch-test-inference-global/run.sh
set -e

JOB_NAME="video-pmd-spatiotemporal-25km-100km-global-5ch-two-block-hiro-patch-test-inference-global"
CONFIG_FILENAME="video_inference.yaml"
WORKSPACE="ai2/ace"
CLUSTER="ai2/titan"  # b200 -- has headroom now; see header comment for fallback
N_GPUS=4
CHECKPOINT_DATASET="01M26N8CBHE5DB0KG08WPR5EWV"
# No WANDB_API_KEY secret needed -- this config has log_to_wandb: false.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_PATH="${SCRIPT_DIR#"$REPO_ROOT"/}/$CONFIG_FILENAME"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run --allow-dirty \
    --name "$JOB_NAME" \
    --description 'Video PMD spatiotemporal TWO-BLOCK (pinned coarse-temporal r + unpinned fine-detail d, fixed kernels), trained at the HiRO-ACE coarse patch size 16x16, test-set inference, GLOBAL DOMAIN in one job via patch-tiled divide_generation, 4-member ensemble, coarse-endpoints-only input, 5 channels, 25km/100km. Checkpoint from a manually stopped run at epoch ~55-60/200 (matching the 44x72 comparison checkpoint depth). ~250 patches/domain (vs. 20 for the 44x72 checkpoint) -- expect longer than that run'"'"'s ~4.9 GPU-day estimate. 4x GPU DDP on titan (b200).' \
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
