#!/bin/bash
# Full GPU training run for the SINGLE-STAGE spatiotemporal video PMD
# trainer, TRUE LR-endpoints-in/HR-full-out variant (flat/independent
# noise), via gantry + torchrun DDP. Reads data from WEKA (climate-default).
#
# Supersedes the stopped
# ../2026-08-05-video-pmd-spatiotemporal-25km-100km-singlestage-v2/run.sh
# (which required real fine-resolution truth at the endpoints, not
# deployable) -- see video_train.yaml's header comment for the architecture
# fix. FRESH start, not resumed: in_channels changed.
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
# Run:  bash configs/experiments/2026-08-07-video-pmd-spatiotemporal-25km-100km-singlestage-coarse-endpoints-flat/run.sh
set -e

JOB_NAME="video-pmd-spatiotemporal-25km-100km-global-5ch-singlestage-coarse-endpoints-flat"
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
    --description 'Spatiotemporal video PMD, SINGLE-STAGE, TRUE LR-endpoints-in/HR-full-out (endpoints_observed=false, coarse_endpoints_only=true -- coarse only at the two endpoint frames, nothing pinned, single joint network diffuses every frame). 5 channels, flat/independent noise, global, patch-trained. 4x GPU DDP on titan (weka). Fresh run.' \
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
