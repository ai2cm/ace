#!/bin/bash
# Full GPU training run for the TWO-BLOCK + CONDITIONAL-KERNEL spatiotemporal
# video PMD trainer (VideoDiffusionModelConfig.two_block_conditional_kernel),
# via gantry + torchrun DDP. Reads data from WEKA (climate-default).
#
# Same LR-endpoints-in/HR-full-out setting, data, and backbone size as
# ../2026-08-26-video-pmd-spatiotemporal-25km-100km-two-block/ (the
# fixed-kernel two-block run) -- see video_train.yaml's header comment for
# what conditional-kernel mode changes on top of that. FRESH start, not
# resumed: this adds the condition-encoder submodules (r_encoder/d_encoder)
# to the checkpoint, which a fixed-kernel checkpoint doesn't have.
#
# Data on weka (/climate-default/2026-06-25-temporal-diffusion/):
#   2026-07-14-X-SHiELD-AMIP-FME-3h-25km.zarr  (fine)
#   2026-07-14-X-SHiELD-AMIP-FME-3h-100km.zarr (coarse)
#
# Prereqs (one-time, PER WORKSPACE -- secrets are workspace-scoped; same
# ones already used by the other PMD run.sh scripts in ai2/climate-titan):
#   pip install beaker-gantry
#   also commit + push your code: gantry runs your pushed git commit.
#
# Run:  bash configs/experiments/2026-08-26-video-pmd-spatiotemporal-25km-100km-two-block-conditional-kernel/run.sh
set -e

JOB_NAME="video-pmd-spatiotemporal-25km-100km-global-5ch-two-block-conditional-kernel"
CONFIG_FILENAME="video_train.yaml"
WORKSPACE="ai2/climate-titan"
CLUSTER="ai2/titan"
N_GPUS=4                             # batch_size in the config must stay divisible by this
WANDB_SECRET="CHLOE_WANDB_API_KEY"   # beaker secret name (in WORKSPACE) holding your W&B key

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_PATH="${SCRIPT_DIR#"$REPO_ROOT"/}/$CONFIG_FILENAME"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run --allow-dirty \
    --name "$JOB_NAME" \
    --description 'Spatiotemporal video PMD, TWO-BLOCK + CONDITIONAL-KERNEL mode (idea/conditional_kernel_theory.md: fixed kernel basis per block + tiny g_phi(c) -> simplex weights, trained by convex moment-matching, not DSM). Same LR-endpoints-in/HR-full-out setting as the two-block fixed-kernel config. 5 channels, global, patch-trained. 4x GPU DDP on titan (weka). Fresh run.' \
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
