#!/bin/bash
# Full GPU training run for the spatiotemporal video PMD trainer (OU
# per-channel noise kernel variant), via gantry + torchrun DDP. Reads data
# DIRECTLY FROM GCS (gs://vcm-ml-intermediate/...), not weka -- the new
# paired 25km/100km dataset isn't mirrored to weka yet.
#
# Cluster/workspace: ai2/titan + ai2/climate-titan, same as the flat variant
# (../2026-07-24-video-pmd-spatiotemporal-25km-100km/run.sh) -- see that
# script's comments for the full cluster-choice/GCS-egress history.
#
# Data on GCS:
#   gs://vcm-ml-intermediate/2026-07-14-X-SHiELD-AMIP-FME-3h-25km.zarr  (fine)
#   gs://vcm-ml-intermediate/2026-07-14-X-SHiELD-AMIP-FME-3h-100km.zarr (coarse)
#
# Prereqs (one-time, PER WORKSPACE -- secrets are workspace-scoped; confirmed
# 2026-07-24 that CHLOE_WANDB_API_KEY and google-credentials both already
# exist in ai2/climate-titan via `beaker secret list -w ai2/climate-titan`):
#   pip install beaker-gantry
#   also commit + push your code: gantry runs your pushed git commit.
#
# Run:  bash configs/experiments/2026-07-27-video-pmd-spatiotemporal-25km-100km-ou/run.sh
#
# Diagnostics-only env vars (carried over from the flat variant's
# 2026-07-25 debugging, kept here too): TORCH_NCCL_TRACE_BUFFER_SIZE /
# TORCH_DISTRIBUTED_DEBUG=DETAIL don't change what's computed, only add a
# small constant overhead per collective op, in exchange for a real stack
# trace if a collective ever desyncs again instead of a silent 30min hang.
set -e

JOB_NAME="video-pmd-spatiotemporal-25km-100km-global-5ch-ou-v1"
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

gantry run \
    --name "$JOB_NAME" \
    --description 'Spatiotemporal video PMD: stage-1 temporal infilling + stage-2 spatial downscaling (25km/100km), 5 channels, OU per-channel noise kernel, no subset training, global, patch-trained. 4x GPU DDP on titan (GCS-direct).' \
    --workspace "$WORKSPACE" \
    --priority urgent \
    --cluster "$CLUSTER" \
    --beaker-image "$DEPS_ONLY_IMAGE" \
    --gpus "$N_GPUS" \
    --shared-memory 64GiB \
    --budget ai2/atec-climate \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY="$WANDB_SECRET" \
    --env TORCH_NCCL_TRACE_BUFFER_SIZE=2000 \
    --env TORCH_DISTRIBUTED_DEBUG=DETAIL \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.downscaling.video_train "$CONFIG_PATH"
