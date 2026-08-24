#!/bin/bash
# Cascade pipeline test: 100km temporal infill (already run) -> 25km spatial
# SR (this job). See inference.yaml's header for the full rationale and
# which channels/models were picked and why.
#
# Checkpoint dataset (SR model, training job v6, 01KZ1WGMKRX6AKWWQXB8JJWRZP,
# succeeded): 01KZ1WGMM2SQR6WSV0R3F3EM47
#
# Infill input (already generated, no job needed here):
#   /climate-default/2026-06-25-temporal-diffusion/inference/
#     video-pmd-5ch-per-channel-kernel-global-1degree-24to3-v1/test-2023-2024-ens32.zarr
#
# Output: /climate-default/2026-06-25-temporal-diffusion/inference/hiro-downscaling-25km-100km-global-5ch-v6-cascade-infill-then-sr/test-2023-2024-ens4.zarr
#
# Prereqs (one-time, PER WORKSPACE -- secrets are workspace-scoped):
#   pip install beaker-gantry
#   also commit + push your code: gantry runs your pushed git commit.
#
# Run:  bash configs/experiments/2026-08-24-hiro-downscaling-cascade-infill-then-sr/run.sh
set -e

JOB_NAME="hiro-downscaling-25km-100km-global-5ch-v6-cascade-infill-then-sr"
CONFIG_FILENAME="inference.yaml"
WORKSPACE="ai2/climate-titan"
CLUSTER="ai2/titan"  # b200
N_GPUS=4
CHECKPOINT_DATASET="01KZ1WGMM2SQR6WSV0R3F3EM47"
WANDB_SECRET="CHLOE_WANDB_API_KEY"
EXPERIMENT_DIR="/climate-default/2026-06-25-temporal-diffusion/inference/hiro-downscaling-25km-100km-global-5ch-v6-cascade-infill-then-sr"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_PATH="${SCRIPT_DIR#"$REPO_ROOT"/}/$CONFIG_FILENAME"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run --allow-dirty \
    --name "$JOB_NAME" \
    --description 'Cascade test: SR model conditioned on the (already-generated) 100km temporal-infill output instead of real dense truth -- realistic sparse-observation deployment scenario. 4x GPU DDP on titan (b200).' \
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
