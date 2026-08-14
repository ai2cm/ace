#!/bin/bash
# Ablation training run: same SINGLE-STAGE spatiotemporal video PMD trainer,
# TRUE LR-endpoints-in/HR-full-out, flat/independent noise, as
# ../2026-08-07-video-pmd-spatiotemporal-25km-100km-singlestage-coarse-endpoints-flat/,
# but with ALL temporal attention removed (see video_train.yaml's header
# comment) -- a HiRO-style, per-frame-independent denoiser that still
# supports coarse-endpoint-only conditioning. Same gantry + torchrun DDP
# pattern. Reads data from WEKA (climate-default).
#
# max_epochs: 31, not 200 -- matches the actual amount of training baked
# into the flat coarse-endpoints model's last checkpoint before it crashed
# (see video_train.yaml's header comment), which is what "comparable" means
# for this ablation.
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
# Run:  bash configs/experiments/2026-08-14-video-pmd-spatiotemporal-25km-100km-singlestage-coarse-endpoints-flat-no-temporal-attn/run.sh
set -e

JOB_NAME="video-pmd-spatiotemporal-25km-100km-global-5ch-singlestage-coarse-endpoints-flat-no-temporal-attn"
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
    --description 'Ablation of the SINGLE-STAGE coarse-endpoints-flat spatiotemporal video PMD model: zero temporal attention anywhere (temporal_attention_levels: []), a HiRO-style per-frame-independent denoiser that still supports coarse-endpoint-only conditioning via the linear-interpolation baseline. 31 epochs (matches the source models actual completed training). 5 channels, flat/independent noise, global, patch-trained. 4x GPU DDP on titan (weka). Fresh run.' \
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
