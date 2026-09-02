#!/bin/bash
# Test-set inference for video-pmd-5ch-flat-global-1degree-24to3-v1 from
# checkpoints/best.ckpt (NOT latest.ckpt) -- see video_inference.yaml's header
# for why (flat's val loss drifted up ~0.009 after ~epoch 160; best.ckpt is
# the fair, training-matched checkpoint vs. the per-channel-OU run).
#
# 32-member ensemble, full held-out test period (2023-01-01 .. 2024-01-04),
# global 1deg, via gantry + torchrun DDP on ai2/titan (4x B200), workspace
# ai2/ace. Reads data and writes the output zarr on weka (climate-default);
# reads the trained checkpoint from its Beaker result dataset.
#
# NB: needs B200 (titan). A first attempt on ai2/jupiter (H100, 80 GiB) OOM'd
# -- this config's batch_size: 8 was tuned for the B200 the original
# 2026-07-22 latest.ckpt run used. Keep it on titan.
#
# Checkpoint dataset (SECOND job under training experiment
# 01KY0SQW5SFS5KEYZR94T6WDTZ; contains checkpoints/best.ckpt written
# 2026-07-21, the "Saving best checkpoint" snapshot from ~epoch 150-160):
#   01KY0V8ZNN763G59S8QBY4304B
#
# Output (sibling of the latest.ckpt run's zarr, -bestckpt suffix):
#   /climate-default/2026-06-25-temporal-diffusion/inference/video-pmd-5ch-flat-global-1degree-24to3-v1/test-2023-2024-ens32-bestckpt.zarr
#
# Expect ~16.5h (ensemble_chunk_size=1, 4 GPUs) -- same as the latest.ckpt run
# (../2026-07-22-video-pmd-5ch-flat-test-inference/).
#
# Prereqs: beaker-gantry installed; code committed + pushed (gantry runs your
# pushed git commit; --allow-dirty lets the untracked eval scratch coexist).
#
# Run:  bash configs/experiments/2026-09-01-video-pmd-5ch-flat-bestckpt-test-inference/run.sh
set -e

JOB_NAME="video-pmd-5ch-flat-global-1degree-24to3-v1-test-inference-bestckpt"
CONFIG_FILENAME="video_inference.yaml"
WORKSPACE="ai2/ace"
CLUSTER="ai2/titan"  # b200 -- H100 OOMs at this config's batch_size
N_GPUS=4
CHECKPOINT_DATASET="01KY0V8ZNN763G59S8QBY4304B"
# No WANDB_API_KEY secret in ai2/ace -- fine, this config has log_to_wandb: false.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
CONFIG_PATH="${SCRIPT_DIR#"$REPO_ROOT"/}/$CONFIG_FILENAME"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

gantry run --allow-dirty \
    --name "$JOB_NAME" \
    --description 'Video PMD test-set inference from best.ckpt (32-member ensemble, flat/independent noise, 5 channels incl. T2m), global 1deg 24h->3h, 4x B200 DDP on titan. Training-matched-checkpoint rerun of the 2026-07-22 latest.ckpt flat inference.' \
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
