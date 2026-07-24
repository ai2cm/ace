#!/bin/bash
# Full GPU training run for the spatiotemporal video PMD trainer, via gantry +
# torchrun DDP. Reads data DIRECTLY FROM GCS (gs://vcm-ml-intermediate/...),
# not weka -- the new paired 25km/100km dataset isn't mirrored to weka yet.
#
# Cluster/workspace: ai2/titan + ai2/climate-titan, per user instruction
# (2026-07-24). NOTE: an earlier version of this script used
# ai2/augusta-google-1 on the theory (from
# ../../baselines/downscaling/run-train-augusta.sh's own comment) that titan
# lacks GCS access; augusta wasn't in this account's allowed-cluster list, so
# that's moot for now, but if the training job itself fails at data-loading
# time (rather than at gantry's cluster-validation step) with GCS connection
# errors, that's the likely cause -- worth confirming titan's egress with
# whoever administers it before spending a long run's budget.
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
# Run:  bash configs/experiments/2026-07-24-video-pmd-spatiotemporal-25km-100km/run.sh
set -e

JOB_NAME="video-pmd-spatiotemporal-25km-100km-global-5ch-per-channel-kernel-subset-cons0-v1"
CONFIG_FILENAME="video_train.yaml"
WORKSPACE="ai2/climate-titan"
CLUSTER="ai2/titan"
N_GPUS=8                             # batch_size in the config must stay divisible by this
WANDB_SECRET="CHLOE_WANDB_API_KEY"   # beaker secret name (in WORKSPACE) holding your W&B key

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_PATH="${SCRIPT_DIR#"$REPO_ROOT"/}/$CONFIG_FILENAME"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run \
    --name "$JOB_NAME" \
    --description 'Spatiotemporal video PMD: stage-1 temporal infilling + stage-2 spatial downscaling (25km/100km), 5 channels, global, patch-trained. 8x GPU DDP on titan (GCS-direct).' \
    --workspace "$WORKSPACE" \
    --priority high \
    --cluster "$CLUSTER" \
    --beaker-image "$DEPS_ONLY_IMAGE" \
    --gpus "$N_GPUS" \
    --shared-memory 64GiB \
    --budget ai2/atec-climate \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY="$WANDB_SECRET" \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.downscaling.video_train "$CONFIG_PATH"
