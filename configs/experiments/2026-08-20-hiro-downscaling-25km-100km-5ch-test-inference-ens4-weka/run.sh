#!/bin/bash
# Rerun of ../2026-08-17-hiro-downscaling-25km-100km-5ch-test-inference-ens4/
# with output written DIRECTLY TO WEKA instead of the job's Beaker result
# dataset -- see inference.yaml's header for why (the Beaker-dataset-mount
# route for that run's ~1.86M-tiny-chunk-file output has failed 5+ times
# downstream trying to mount it for CRPS/diurnal/TC verification).
#
# Checkpoint dataset (training job v6, 01KZ1WGMKRX6AKWWQXB8JJWRZP, succeeded):
#   01KZ1WGMM2SQR6WSV0R3F3EM47
#
# Output: /climate-default/2026-06-25-temporal-diffusion/inference/hiro-downscaling-25km-100km-global-5ch-v6/test-2023-2024-ens4.zarr
#
# Prereqs (one-time, PER WORKSPACE -- secrets are workspace-scoped):
#   pip install beaker-gantry
#   also commit + push your code: gantry runs your pushed git commit.
#
# Run:  bash configs/experiments/2026-08-20-hiro-downscaling-25km-100km-5ch-test-inference-ens4-weka/run.sh
set -e

JOB_NAME="hiro-downscaling-25km-100km-global-5ch-v6-test-inference-ens4-weka"
CONFIG_FILENAME="inference.yaml"
WORKSPACE="ai2/climate-titan"
CLUSTER="ai2/titan"  # b200
N_GPUS=4
CHECKPOINT_DATASET="01KZ1WGMM2SQR6WSV0R3F3EM47"
WANDB_SECRET="CHLOE_WANDB_API_KEY"
EXPERIMENT_DIR="/climate-default/2026-06-25-temporal-diffusion/inference/hiro-downscaling-25km-100km-global-5ch-v6"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_PATH="${SCRIPT_DIR#"$REPO_ROOT"/}/$CONFIG_FILENAME"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run --allow-dirty \
    --name "$JOB_NAME" \
    --description 'HiRO-ACE-style spatial downscaling (100km->25km, 5ch) full test-set frame-by-frame inference, global patch-tiled, n_ens=4, output written directly to weka instead of a Beaker result dataset (see inference.yaml header). 4x GPU DDP on titan (b200).' \
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
    -- bash -c "mkdir -p '$EXPERIMENT_DIR' && torchrun --nproc_per_node $N_GPUS -m fme.downscaling.inference '$CONFIG_PATH'"
