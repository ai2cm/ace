#!/bin/bash
# Full GPU training run for the HiRO-ACE-style spatial downscaling baseline
# (100km -> 25km, factor 4, 5 channels), via gantry + torchrun DDP. Ported
# from Anna Kwa's config/global-moe-models config (see train.yaml's header)
# -- data already lives on WEKA under climate-default (no GCS copy needed
# for this version, unlike the from-scratch config this replaced), so the
# existing --weka mount below covers it as-is.
#
# Uses fme.downscaling.train (the plain spatial trainer), NOT
# fme.downscaling.video_train -- this is a single-frame baseline, not PMD.
#
# Prereqs (one-time, PER WORKSPACE -- secrets are workspace-scoped; same
# ones already used by the PMD run.sh scripts in ai2/climate-titan):
#   pip install beaker-gantry
#   also commit + push your code: gantry runs your pushed git commit.
#
# Run:  bash configs/experiments/2026-07-28-hiro-downscaling-25km-100km-5ch/run.sh
set -e

JOB_NAME="hiro-downscaling-25km-100km-global-5ch-v3-moe-port"
CONFIG_FILENAME="train.yaml"
WORKSPACE="ai2/climate-titan"
CLUSTER="ai2/titan"
N_GPUS=4                             # batch_size in the config must stay divisible by this
WANDB_SECRET="CHLOE_WANDB_API_KEY"   # beaker secret name (in WORKSPACE) holding your W&B key

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_PATH="${SCRIPT_DIR#"$REPO_ROOT"/}/$CONFIG_FILENAME"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run \
    --name "$JOB_NAME" \
    --description 'HiRO-ACE-style spatial downscaling baseline (100km->25km, 5ch), ported from config/global-moe-models, unet_diffusion_song_v2, global, patch-trained. 4x GPU DDP on titan (weka).' \
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
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.downscaling.train "$CONFIG_PATH"
