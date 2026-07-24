#!/bin/bash
# Full GPU training run for the spatiotemporal video PMD trainer, via gantry +
# torchrun DDP. Reads data DIRECTLY FROM GCS (gs://vcm-ml-intermediate/...),
# not weka -- the new paired 25km/100km dataset isn't mirrored to weka yet.
#
# Cluster: uses ai2/augusta-google-1, NOT titan. Titan (used by the stage-1
# video PMD configs, e.g. ../2026-07-20-video-pmd-5ch-per-channel-kernel-subset-cons0/)
# does not have GCS access; augusta does but not weka -- see
# ../../baselines/downscaling/run-train-augusta.sh, the equivalent script for
# HiRO's own 2D (spatial-only) downscaling training, which this mirrors.
#
# Data on GCS:
#   gs://vcm-ml-intermediate/2026-07-14-X-SHiELD-AMIP-FME-3h-25km.zarr  (fine)
#   gs://vcm-ml-intermediate/2026-07-14-X-SHiELD-AMIP-FME-3h-100km.zarr (coarse)
#
# Prereqs (one-time, PER WORKSPACE -- secrets are workspace-scoped):
#   pip install beaker-gantry
#   beaker secret write --workspace <WORKSPACE> CHLOE_WANDB_API_KEY <your-wandb-key>
#   # a GCS service-account key readable by the training job, as a beaker
#   # dataset-secret named "google-credentials" in <WORKSPACE> (see
#   # ../../baselines/downscaling/run-train-augusta.sh and
#   # ../../baselines/era5/run-ace-train.sh for the same pattern) -- ask
#   # whoever manages ai2/downscaling or ai2/climate-titan's secrets if this
#   # doesn't already exist in your workspace.
#   # also commit + push your code: gantry runs your pushed git commit.
#
# Run:  bash configs/experiments/2026-07-24-video-pmd-spatiotemporal-25km-100km/run.sh
set -e

JOB_NAME="video-pmd-spatiotemporal-25km-100km-global-5ch-per-channel-kernel-subset-cons0-v1"
CONFIG_FILENAME="video_train.yaml"
WORKSPACE="ai2/downscaling"          # CONFIRM: matches the GCS-direct HiRO downscaling jobs' workspace, NOT ai2/climate-titan (weka-based)
CLUSTER="ai2/augusta-google-1"       # has GCS access, no weka -- see comment above
N_GPUS=8                             # batch_size in the config must stay divisible by this
WANDB_SECRET="CHLOE_WANDB_API_KEY"   # beaker secret name (in WORKSPACE) holding your W&B key

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_PATH="${SCRIPT_DIR#"$REPO_ROOT"/}/$CONFIG_FILENAME"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run \
    --name "$JOB_NAME" \
    --description 'Spatiotemporal video PMD: stage-1 temporal infilling + stage-2 spatial downscaling (25km/100km), 5 channels, global, patch-trained. 8x GPU DDP on augusta (GCS-direct).' \
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
