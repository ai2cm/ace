#!/bin/bash
# Lo-only (from-noise@200) ABLATION eval — is Student-Hi worth keeping?
#
# Evaluates the SINGLE-MODEL Student-Lo checkpoint alone (expert 0,
# best_student_tail.ckpt — the SAME ckpt bundled into the combined [Hi->Lo]
# student rmoodemk), with its noise schedule capped at sigma_max=200 so it samples
# from fresh noise@200 straight to x0 (config-lo-only.yaml handles the model_updates).
# CONUS 2023 holdout, 3 km X-SHiELD AMIP — identical data/patch/n_samples/events to
# config-distilled.yaml, so the wandb metrics (project andrep-downscaling) are
# directly comparable to the combined bundle eval rmoodemk and the teacher 1r1p6djp.
#
# Compare afterwards with:
#   check_runs.py --compare-eval rmoodemk <this-run> --project andrep-downscaling
# Decision: if Lo-only ~= the bundle (esp. coarse/PRMSL/low-k), Hi adds no utility
# and can be dropped (fewer params + one fewer NFE). See the write-up:
#   fme/downscaling/distillation/experiments/reports/2026-07-13-lo-only-from-noise200-ablation-TBD.md
#
# Usage: ./run-lo-only.sh [--suffix <suffix>]
set -e

BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR#$REPO_ROOT/}"
cd "$REPO_ROOT"

NGPU=4
IMAGE="$(cat "$REPO_ROOT/latest_deps_only_image.txt")"

# Student-Lo checkpoint dataset (single-net .ckpt written by the training callback).
# Mount ONLY the student_checkpoints subpath at /lo, matching how run.sh's bundle
# mode references it (so config-lo-only.yaml finds /lo/best_student_tail.ckpt).
DATASET_LO=01KWJAFM694MAE55M2JMZSE89M   # …-baseline-fixed-moe-teacher-expert0
RUN_LO=ace-downscaling-distillation-fdistill-with-val-baseline-fixed-moe-teacher-expert0

SUFFIX=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suffix)
            [[ -z "${2:-}" ]] && { echo "Error: --suffix requires a value"; exit 1; }
            SUFFIX="-$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

JOB_NAME="evaluate-moe-lo-only-from-noise200-xshield-amip-100km-to-3km-conus${SUFFIX}"

gantry run \
    --name "$JOB_NAME" \
    --description "Lo-only (from-noise@200) ablation: single-model Student-Lo (beaker://${DATASET_LO}, best_student_tail.ckpt, sigma_max=200, 2-step) on CONUS 2023 holdout (3 km). Compare vs combined [Hi->Lo] bundle rmoodemk to decide if Hi is droppable." \
    --workspace ai2/climate-titan \
    --priority urgent \
    --preemptible \
    --cluster ai2/titan \
    --beaker-image "$IMAGE" \
    --env WANDB_USERNAME="$BEAKER_USERNAME" \
    --env WANDB_NAME="$JOB_NAME" \
    --env WANDB_JOB_TYPE=inference \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "${DATASET_LO}:fastgen/${RUN_LO}/student_checkpoints:/lo" \
    --weka climate-default:/climate-default \
    --gpus $NGPU \
    --shared-memory 400GiB \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node $NGPU -m fme.downscaling.evaluator "$SCRIPT_PATH/config-lo-only.yaml"
