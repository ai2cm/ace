#!/bin/bash
# Resumes ../2026-07-28-hiro-downscaling-25km-100km-5ch/ for another 200
# epochs (200 -> 400 total), to test whether HiRO's gap vs. the video-PMD
# models (toy/single_vs_two_stage_v1_comparison.md) narrows with more
# training. Same gantry + torchrun DDP pattern as the source run, plus a
# --dataset mount of the prior run's checkpoints for train.yaml's
# resume_results_dir.
#
# Checkpoint dataset (result of the finished 200-epoch run, experiment
# 01KZ1WGMKRX6AKWWQXB8JJWRZP / wandb ai2cm/multivariate-downscaling/gbm6d4ag):
CHECKPOINT_DATASET="01KZ1WGMM2SQR6WSV0R3F3EM47"
#
# The source run took ~26h wall-clock for 200 epochs on 4x GPU -- budget
# for a similar duration here.
#
# Prereqs (one-time, PER WORKSPACE -- secrets are workspace-scoped; same
# ones already used by the PMD run.sh scripts in ai2/climate-titan):
#   pip install beaker-gantry
#   also commit + push your code: gantry runs your pushed git commit.
#
# Run:  bash configs/experiments/2026-08-13-hiro-downscaling-25km-100km-5ch-extend-400/run.sh
set -e

JOB_NAME="hiro-downscaling-25km-100km-global-5ch-v6-extend-400"
CONFIG_FILENAME="train.yaml"
WORKSPACE="ai2/climate-titan"
CLUSTER="ai2/titan"
N_GPUS=4                             # batch_size in the config must stay divisible by this
WANDB_SECRET="CHLOE_WANDB_API_KEY"   # beaker secret name (in WORKSPACE) holding your W&B key
# Same rationale as the source config: full-domain validation-with-generation
# takes ~4h/pass, comfortably over the 30-min default NCCL collective
# timeout. Paired with train.yaml's validate_interval: 50.
NCCL_TIMEOUT_MINUTES=360

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_PATH="${SCRIPT_DIR#"$REPO_ROOT"/}/$CONFIG_FILENAME"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run --allow-dirty \
    --name "$JOB_NAME" \
    --description 'Resume of hiro-downscaling-25km-100km-global-5ch-v6-validation-fix for another 200 epochs (200->400 total), to test whether more training narrows the gap vs. the video-PMD models.' \
    --workspace "$WORKSPACE" \
    --priority urgent \
    --cluster "$CLUSTER" \
    --beaker-image "$DEPS_ONLY_IMAGE" \
    --gpus "$N_GPUS" \
    --shared-memory 64GiB \
    --budget ai2/atec-climate \
    --weka climate-default:/climate-default \
    --dataset "${CHECKPOINT_DATASET}:/resume_results" \
    --env-secret WANDB_API_KEY="$WANDB_SECRET" \
    --env FME_DISTRIBUTED_TIMEOUT_MINUTES="$NCCL_TIMEOUT_MINUTES" \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.downscaling.train "$CONFIG_PATH"
