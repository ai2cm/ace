#!/bin/bash
# Test-set inference for global-1degree-24to3-pcn-v1: 32-member ensemble
# infilling over the held-out test period, via gantry + torchrun DDP on
# ai2/titan (4x B200). Reads data from weka (climate-default), reads the
# trained checkpoint from its Beaker result dataset, writes the output zarr
# back to weka.
#
# Checkpoint dataset (result of the training job, contains checkpoints/best.ckpt):
#   01KWDBK1BTEPRP6K7WRKVD1826
#
# Output:
#   /climate-default/2026-06-25-temporal-diffusion/inference/global-1degree-24to3-pcn-v1/test-2023-2024-ens32.zarr
#
# Prereqs (one-time, PER WORKSPACE -- secrets are workspace-scoped):
#   pip install beaker-gantry
#   beaker secret write --workspace ai2/climate-titan CHLOE_WANDB_API_KEY <your-wandb-key>
#   # also commit + push your code: gantry runs your pushed git commit.
#
# Smoke test first (small ensemble, capped batches) before the full run:
#   Edit video_inference.yaml: n_ensemble: 2, add `max_batches: 4`, then run
#   this script pointed at a scratch output_path. Confirms the checkpoint
#   loads and ensemble_chunk_size fits in GPU memory before committing to the
#   full test period.
#
# Run:  bash configs/experiments/2026-07-14-video-pmd-pcn-test-inference/run.sh
set -e

JOB_NAME="video-pmd-pcn-global-1degree-24to3-v1-test-inference"
CONFIG_FILENAME="video_inference.yaml"
WORKSPACE="ai2/climate-titan"
CLUSTER="ai2/titan"
N_GPUS=4
CHECKPOINT_DATASET="01KWDBK1BTEPRP6K7WRKVD1826"
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
    --description 'Video PMD test-set inference (32-member ensemble), global 1deg 24h->3h, 4x B200 DDP' \
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
