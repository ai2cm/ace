#!/bin/bash
# Full GPU training run for the spatiotemporal video PMD trainer, via gantry +
# torchrun DDP. Reads data from WEKA (climate-default) as of 2026-07-29 --
# both the 25km and 100km zarrs were copied there from GCS for cost
# efficiency. Also RESUMES from this job's own previous run (killed
# 2026-07-29 by request), mounting that run's result dataset
# (01KYGFN978NFP253PY41YWHJ1Z) read-only and pointing
# resume_results_dir at it -- see video_train.yaml's header for details.
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
# Data on weka (/climate-default/2026-06-25-temporal-diffusion/):
#   2026-07-14-X-SHiELD-AMIP-FME-3h-25km.zarr  (fine)
#   2026-07-14-X-SHiELD-AMIP-FME-3h-100km.zarr (coarse)
#
# Prereqs (one-time, PER WORKSPACE -- secrets are workspace-scoped; confirmed
# 2026-07-24 that CHLOE_WANDB_API_KEY already exists in ai2/climate-titan via
# `beaker secret list -w ai2/climate-titan`; google-credentials no longer
# needed since data moved to weka):
#   pip install beaker-gantry
#   also commit + push your code: gantry runs your pushed git commit.
#
# Run:  bash configs/experiments/2026-07-24-video-pmd-spatiotemporal-25km-100km/run.sh
#
# Diagnostics-only env vars (2026-07-25): the previous attempt hung on an
# NCCL ALLREDUCE for 30min (rank3 desynced from ranks 0/1/2) with no stack
# trace, since the flight recorder was off. TORCH_NCCL_TRACE_BUFFER_SIZE /
# TORCH_DISTRIBUTED_DEBUG=DETAIL just make a repeat hang debuggable -- they
# don't change what's computed, so they don't affect training results, only
# add a small constant overhead per collective op (DETAIL adds
# consistency-checking logging; the trace buffer is a small ring buffer of
# recent collective ops). Safe to leave on; remove once the hang is
# understood/resolved if you want to shave off the (minor) overhead.
set -e

JOB_NAME="video-pmd-spatiotemporal-25km-100km-global-5ch-flat-v2-resume"
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
    --description 'Spatiotemporal video PMD: stage-1 temporal infilling + stage-2 spatial downscaling (25km/100km), 5 channels, flat/independent noise, no subset training, global, patch-trained. 4x GPU DDP on titan (weka), resumed from prior killed run.' \
    --workspace "$WORKSPACE" \
    --priority urgent \
    --cluster "$CLUSTER" \
    --beaker-image "$DEPS_ONLY_IMAGE" \
    --gpus "$N_GPUS" \
    --shared-memory 64GiB \
    --budget ai2/atec-climate \
    --weka climate-default:/climate-default \
    --dataset 01KYGFN978NFP253PY41YWHJ1Z:/resume_results \
    --env-secret WANDB_API_KEY="$WANDB_SECRET" \
    --env TORCH_NCCL_TRACE_BUFFER_SIZE=2000 \
    --env TORCH_DISTRIBUTED_DEBUG=DETAIL \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.downscaling.video_train "$CONFIG_PATH"
