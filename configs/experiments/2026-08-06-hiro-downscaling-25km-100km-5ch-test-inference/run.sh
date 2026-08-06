#!/bin/bash
# Full test-set frame-by-frame inference for hiro-downscaling-25km-100km-global-5ch-v6:
# global (patch-tiled, not regionally split -- see inference.yaml header),
# single-realization (n_ens=1) generation over the full held-out test period
# (2023-01-01 .. 2024-01-04), via gantry + torchrun DDP on ai2/titan. Reads
# data from weka, reads the trained checkpoint from its Beaker result
# dataset, writes the output zarr to the job's /results (becomes a Beaker
# result dataset -- no shared-weka write needed).
#
# Checkpoint dataset (training job v6, 01KZ1WGMKRX6AKWWQXB8JJWRZP, succeeded):
#   01KZ1WGMM2SQR6WSV0R3F3EM47
#
# Output: {result dataset}/test-2023-2024.zarr, dims (time, ensemble, lat, lon)
#
# Prereqs (one-time, PER WORKSPACE -- secrets are workspace-scoped):
#   pip install beaker-gantry
#   also commit + push your code: gantry runs your pushed git commit.
#
# Run:  bash configs/experiments/2026-08-06-hiro-downscaling-25km-100km-5ch-test-inference/run.sh
set -e

JOB_NAME="hiro-downscaling-25km-100km-global-5ch-v6-test-inference"
CONFIG_FILENAME="inference.yaml"
WORKSPACE="ai2/climate-titan"
CLUSTER="ai2/titan"
N_GPUS=4
CHECKPOINT_DATASET="01KZ1WGMM2SQR6WSV0R3F3EM47"
WANDB_SECRET="CHLOE_WANDB_API_KEY"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_PATH="${SCRIPT_DIR#"$REPO_ROOT"/}/$CONFIG_FILENAME"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run \
    --name "$JOB_NAME" \
    --description 'HiRO-ACE-style spatial downscaling (100km->25km, 5ch) full test-set frame-by-frame inference, global patch-tiled, n_ens=1, using fme.downscaling.inference (production tool with real patch compositing). 4x GPU DDP on titan.' \
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
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.downscaling.inference "$CONFIG_PATH"
