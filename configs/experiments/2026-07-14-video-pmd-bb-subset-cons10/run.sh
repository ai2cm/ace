#!/bin/bash
# Full GPU training run for the video PMD trainer on ai2/titan (4x B200), via
# gantry + torchrun DDP. Reads data from weka (climate-default).
#
# BB + per-channel noise + subset training + marginal-consistency loss
# (weight 10). Third arm of the cons0 / cons1 / cons10 L_marg sweep.
#
# Data on weka:
#   /climate-default/2026-06-25-temporal-diffusion/2025-07-25-X-SHiELD-AMIP-FME-3h.zarr
#
# Prereqs (one-time, PER WORKSPACE -- secrets are workspace-scoped):
#   pip install beaker-gantry
#   beaker secret write --workspace ai2/climate-titan CHLOE_WANDB_API_KEY <your-wandb-key>
#   # also commit + push your code: gantry runs your pushed git commit.
#
# Run:  bash configs/experiments/2026-07-14-video-pmd-bb-subset-cons10/run.sh
set -e

JOB_NAME="video-pmd-bb-pcn-subset-cons10-global-1degree-24to3-v1"
CONFIG_FILENAME="video_train.yaml"
WORKSPACE="ai2/climate-titan"
CLUSTER="ai2/titan"
N_GPUS=4                              # batch_size in the config must stay divisible by this
WANDB_SECRET="CHLOE_WANDB_API_KEY"   # beaker secret name (in WORKSPACE) holding your W&B key

# Resolve the config path relative to the repo root from the script's OWN
# location, so this works whether invoked from the repo root or the script dir.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_PATH="${SCRIPT_DIR#"$REPO_ROOT"/}/$CONFIG_FILENAME"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run \
    --name "$JOB_NAME" \
    --description 'Video PMD (BB + per-channel noise + subset training + L_marg weight 10), global 1deg 24h->3h, 4x B200 DDP' \
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
