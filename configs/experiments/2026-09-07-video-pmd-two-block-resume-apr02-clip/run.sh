#!/bin/bash
# Targeted re-run of the single missing 2023-04-02 clip for the two-block
# global inference zarr -- see video_inference.yaml header.
#
# SINGLE GPU on purpose: no DDP -> the final-barrier NCCL timeout that killed
# the original 4-GPU run (experiment 01M1HFBRC65YMQ66YDPDHAZ8XX) cannot recur.
# One patch-tiled clip ~= 76 min on one L40s (per-clip rate from the original
# run's ~4.9 GPU-days / 92 clips per rank).
#
# After it finishes, merge the clip into the main store (never overwritten):
#   python configs/experiments/2026-09-07-video-pmd-two-block-resume-apr02-clip/merge_apr02_clip.py
# run from a weka-mounted Beaker session.
#
# Run:  bash configs/experiments/2026-09-07-video-pmd-two-block-resume-apr02-clip/run.sh
set -e

JOB_NAME="video-pmd-two-block-coarse-endpoints-flat-RESUME-apr02-clip"
CONFIG_FILENAME="video_inference.yaml"
WORKSPACE="ai2/ace"
CLUSTER="ai2/neptune"  # l40s -- same hardware the original clips were generated on
N_GPUS=1
CHECKPOINT_DATASET="01M100MWQDFSZHWAQW1ZTJZFJ4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_PATH="${SCRIPT_DIR#"$REPO_ROOT"/}/$CONFIG_FILENAME"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run --allow-dirty \
    --name "$JOB_NAME" \
    --description 'Targeted re-run of the one missing 2023-04-02 tumbling clip for the two-block global inference zarr (original run 01M1HFBRC65YMQ66YDPDHAZ8XX died on a final-barrier NCCL timeout 1 clip short). Single-GPU, max_batches=1, writes to a scratch store; a separate merge step copies the 8-frame region into the main store via zarr r+.' \
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
