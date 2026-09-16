#!/bin/bash
# Test-set inference for
# video-pmd-spatiotemporal-25km-100km-global-5ch-two-block-hiro-patch,
# GLOBAL DOMAIN IN ONE JOB via divide_generation -- see video_inference.yaml's
# header for the full rationale/caveats (checkpoint stopped at epoch ~55-60,
# matching the 44x72 comparison checkpoint's own stop depth; coarse_patch_extent
# 16x16 tiles the domain into ~250 patches instead of 20).
#
# Checkpoint dataset: 01M26N8CBHE5DB0KG08WPR5EWV
#
# Output:
#   /climate-default/2026-06-25-temporal-diffusion/inference/video-pmd-spatiotemporal-25km-100km-global-5ch-two-block-hiro-patch/test-2023-2024-ens4-global.zarr
#
# Cluster: ai2/neptune (L40s), NOT ai2/titan (B200) despite the earlier
# comment here -- switched 2026-09-16 after the first titan attempt
# (01M2GGAVQYTQVHGDC5X91FZ0PC) burned through 3 job attempts over ~2 days
# and made zero net progress: attempt 1 preempted before writing anything;
# attempt 2 got to batch 15/92 over ~19.5h before its NODE WAS CORDONED
# (ops action, not a priority preemption); attempt 3 was itself preempted by
# a higher-priority job 19 min in; a 4th attempt then sat "pending"
# (unscheduled) for 5.7h because titan had dropped to 1/80 free slots with
# 16 GPUs cordoned. EVERY restart re-runs video_inference.py's
# writer.initialize_store(mode="w") unconditionally (no resume-from-
# partial-progress support), so each of those preemptions wiped batch
# progress back to zero -- the cordon/preemption cost ~19.5h of real work,
# not just wall-clock. Neptune had 54/72 free when this was written; its
# "eager" scheduling policy (vs. titan's "strict priority") should also
# mean less preemption churn for a job this long. Re-check
# `beaker cluster get ai2/neptune` if this sits queued.
#
# Prereqs (one-time, PER WORKSPACE -- secrets are workspace-scoped):
#   pip install beaker-gantry
#   also commit + push your code: gantry runs your pushed git commit.
#
# Run:  bash configs/experiments/2026-09-15-video-pmd-spatiotemporal-25km-100km-two-block-hiro-patch-test-inference-global/run.sh
set -e

JOB_NAME="video-pmd-spatiotemporal-25km-100km-global-5ch-two-block-hiro-patch-test-inference-global"
CONFIG_FILENAME="video_inference.yaml"
WORKSPACE="ai2/ace"
CLUSTER="ai2/neptune"  # l40s -- titan was thrashing (see header comment)
N_GPUS=4
CHECKPOINT_DATASET="01M26N8CBHE5DB0KG08WPR5EWV"
# No WANDB_API_KEY secret needed -- this config has log_to_wandb: false.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_PATH="${SCRIPT_DIR#"$REPO_ROOT"/}/$CONFIG_FILENAME"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run --allow-dirty \
    --name "$JOB_NAME" \
    --description 'Video PMD spatiotemporal TWO-BLOCK (pinned coarse-temporal r + unpinned fine-detail d, fixed kernels), trained at the HiRO-ACE coarse patch size 16x16, test-set inference, GLOBAL DOMAIN in one job via patch-tiled divide_generation, 4-member ensemble, coarse-endpoints-only input, 5 channels, 25km/100km. Checkpoint from a manually stopped run at epoch ~55-60/200 (matching the 44x72 comparison checkpoint depth). ~250 patches/domain (vs. 20 for the 44x72 checkpoint) -- expect longer than that run'"'"'s ~4.9 GPU-day estimate. 4x GPU DDP on neptune (l40s) -- moved off titan after repeated preemption/cordon churn wiped progress 3x.' \
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
