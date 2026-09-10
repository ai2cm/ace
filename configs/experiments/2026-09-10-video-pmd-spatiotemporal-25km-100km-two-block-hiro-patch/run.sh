#!/bin/bash
# Full GPU training run for the TWO-BLOCK spatiotemporal video PMD trainer
# (VideoDiffusionModelConfig.two_block) at the HiRO-ACE patch size, via
# gantry + torchrun DDP. Reads data from WEKA (climate-default).
#
# Identical to
# ../2026-08-26-video-pmd-spatiotemporal-25km-100km-two-block/run.sh and
# its video_train.yaml EXCEPT coarse_patch_extent_lat/lon: 44/72 -> 16/16,
# matching the patch size HiRO-ACE trains on
# (../2026-08-13-hiro-downscaling-25km-100km-5ch-extend-400/train.yaml).
# Fixed (flat) kernel: r_kernel brownian_bridge, d_kernel independent --
# NOT the conditional-kernel variant. FRESH start.
#
# Data on weka (/climate-default/2026-06-25-temporal-diffusion/):
#   2026-07-14-X-SHiELD-AMIP-FME-3h-25km.zarr  (fine)
#   2026-07-14-X-SHiELD-AMIP-FME-3h-100km.zarr (coarse)
#
# Runs in the ai2/ace workspace (not ai2/climate-titan like the other PMD
# configs) -- uses that workspace's shared service-account W&B secret
# `wandb-api-key-ai2cm-sa` (the convention for ai2/ace gantry runs; logs to
# the ai2cm entity set in video_train.yaml). Titan is all 8x B200 nodes, so
# --gpus 4 gets 4x B200.
#
# Prereqs:
#   pip install beaker-gantry
#   also commit + push your code: gantry runs your pushed git commit.
#
# Run:  bash configs/experiments/2026-09-10-video-pmd-spatiotemporal-25km-100km-two-block-hiro-patch/run.sh
set -e

JOB_NAME="video-pmd-spatiotemporal-25km-100km-global-5ch-two-block-hiro-patch"
CONFIG_FILENAME="video_train.yaml"
WORKSPACE="ai2/ace"
CLUSTER="ai2/titan"
N_GPUS=4                                   # config batch_size (16) must stay divisible by this
WANDB_SECRET="wandb-api-key-ai2cm-sa"      # beaker secret name in WORKSPACE holding the W&B key

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_PATH="${SCRIPT_DIR#"$REPO_ROOT"/}/$CONFIG_FILENAME"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run --allow-dirty \
    --name "$JOB_NAME" \
    --description 'Spatiotemporal video PMD, TWO-BLOCK mode (fixed/flat kernel: r brownian_bridge, d independent), trained at the HiRO-ACE coarse patch size 16x16 instead of 44x72 -- removes patch-size as a confound in the two-block-vs-HiRO comparison. Batch size raised 4->16 (smaller tiles free up B200 memory). Same data, channels, backbone, and 200-epoch budget as the 20-patch two-block run. 5 channels, global, patch-trained. 4x B200 DDP on titan (weka). Fresh run.' \
    --workspace "$WORKSPACE" \
    --priority urgent \
    --cluster "$CLUSTER" \
    --beaker-image "$DEPS_ONLY_IMAGE" \
    --gpus "$N_GPUS" \
    --shared-memory 64GiB \
    --budget ai2/atec-climate \
    --weka climate-default:/climate-default \
    --env-secret WANDB_API_KEY="$WANDB_SECRET" \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.downscaling.video_train "$CONFIG_PATH"
