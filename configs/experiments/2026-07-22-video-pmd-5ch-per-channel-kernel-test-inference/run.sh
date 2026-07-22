#!/bin/bash
# Test-set inference for video-pmd-5ch-per-channel-kernel-global-1degree-
# 24to3-v1: 32-member ensemble infilling over the full held-out test period
# (2023-01-01 .. 2024-01-04), via gantry + torchrun DDP on ai2/titan (4x
# B200). Reads data from weka (climate-default), reads the trained checkpoint
# from its Beaker result dataset, writes the output zarr back to weka.
#
# Checkpoint dataset (result of the training job, contains checkpoints/best.ckpt;
# training completed cleanly, exitCode 0, 200 epochs, no preemption):
#   01KY0SRVAY199HM7TM8M9TXEVX
#
# Output:
#   /climate-default/2026-06-25-temporal-diffusion/inference/video-pmd-5ch-per-channel-kernel-global-1degree-24to3-v1/test-2023-2024-ens32.zarr
#
# Expect a similar order of magnitude as the earlier 4-channel inference runs
# (~16.5h with ensemble_chunk_size=1, 4 GPUs) -- same architecture family,
# same ensemble size, same full-year test period, one extra channel (T2m).
#
# Prereqs (one-time, PER WORKSPACE -- secrets are workspace-scoped):
#   pip install beaker-gantry
#   beaker secret write --workspace ai2/climate-titan CHLOE_WANDB_API_KEY <your-wandb-key>
#   # also commit + push your code: gantry runs your pushed git commit.
#
# Run:  bash configs/experiments/2026-07-22-video-pmd-5ch-per-channel-kernel-test-inference/run.sh
set -e

JOB_NAME="video-pmd-5ch-per-channel-kernel-global-1degree-24to3-v1-test-inference"
CONFIG_FILENAME="video_inference.yaml"
WORKSPACE="ai2/climate-titan"
CLUSTER="ai2/titan"
N_GPUS=4
CHECKPOINT_DATASET="01KY0SRVAY199HM7TM8M9TXEVX"
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
    --description 'Video PMD test-set inference (32-member ensemble, per-channel OU noise kernel, 5 channels incl. T2m), global 1deg 24h->3h, 4x B200 DDP' \
    --workspace "$WORKSPACE" \
    --priority urgent \
    --cluster "$CLUSTER" \
    --beaker-image "$DEPS_ONLY_IMAGE" \
    --gpus "$N_GPUS" \
    --shared-memory 64GiB \
    --budget ai2/atec-climate \
    --weka climate-default:/climate-default \
    --dataset "${CHECKPOINT_DATASET}:/checkpoint" \
    --env-secret WANDB_API_KEY="$WANDB_SECRET" \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.downscaling.video_inference "$CONFIG_PATH"
