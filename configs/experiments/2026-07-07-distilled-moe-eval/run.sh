#!/bin/bash
# Assemble the distilled 2-step MoE student bundle, then evaluate it vs the
# original bundled MoE teacher on the CONUS 2023 holdout year, against 3 km
# X-SHiELD AMIP ground truth. The two eval configs share identical
# data/patch/n_samples, so the resulting wandb metrics (CRPS / spectra / tails,
# project andrep-downscaling) are directly comparable. This also produces eval
# stats that do not otherwise exist for the bundled teacher.
#
# Usage:
#   ./run.sh bundle                      # assemble the student bundle -> weka
#   ./run.sh <teacher|distilled|all> [--suffix <suffix>]   # evaluate
#
# Run `bundle` first and let it finish before `distilled`/`all` (the eval reads
# the bundle it writes to weka).
set -e

BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR#$REPO_ROOT/}"
cd "$REPO_ROOT"

NGPU=4
IMAGE="$(cat "$REPO_ROOT/latest_deps_only_image.txt")"

# Original bundled multivariate MoE teacher (root mount, no checkpoints/ subdir).
DATASET_TEACHER=01KTCHVDHY0SATWH9E0AW2PDS6

# Per-expert distilled student checkpoint datasets (single-net .ckpt written by
# the training callback). We mount ONLY each dataset's student_checkpoints
# subpath, so unrelated run artifacts are not downloaded.
DATASET_LO=01KWJAFM694MAE55M2JMZSE89M   # …-baseline-fixed-moe-teacher-expert0
DATASET_HI=01KWTXGAM1CCGDH29JWDSN9KPF   # …-hi-1step-moe-teacher-expert1
RUN_LO=ace-downscaling-distillation-fdistill-with-val-baseline-fixed-moe-teacher-expert0
RUN_HI=ace-downscaling-distillation-fdistill-with-val-hi-1step-moe-teacher-expert1

# Where the assembled bundle is written on weka, and read by config-distilled.yaml.
BUNDLE_DIR=/climate-default/2026-07-07-distilled-moe-bundle
BUNDLE_PATH="$BUNDLE_DIR/distilled_moe_bundle.ckpt"

usage() {
    echo "Usage: $0 <bundle|teacher|distilled|all> [--suffix <suffix>]"
    exit 1
}

# Assemble the [2,1] student cascade bundle from the two per-expert checkpoints
# and write it to weka. Subpath mounts keep the download to just the
# student_checkpoints dirs. Source datasets are recorded both as gantry inputs
# (the mounts below) and explicitly in the description.
run_bundle() {
    gantry run \
        --name "bundle-distilled-moe-student" \
        --description "Bundle distilled 2-step MoE student: Lo=expert0 baseline-fixed (beaker://${DATASET_LO}) + Hi=expert1 hi-1step (beaker://${DATASET_HI}); best_student_tail.ckpt each, steps_per_range [2,1]" \
        --workspace ai2/climate-titan \
        --priority normal \
        --cluster ai2/titan \
        --beaker-image "$IMAGE" \
        --dataset "${DATASET_LO}:fastgen/${RUN_LO}/student_checkpoints:/lo" \
        --dataset "${DATASET_HI}:fastgen/${RUN_HI}/student_checkpoints:/hi" \
        --weka climate-default:/climate-default \
        --gpus 1 \
        --shared-memory 100GiB \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- bash -c "mkdir -p '$BUNDLE_DIR' && python scripts/downscaling/bundle_denoising_moe_checkpoint.py '$SCRIPT_PATH/distilled-bundle.yaml' '$BUNDLE_PATH'"
}

run_eval() {
    local model="$1"
    local config="$2"
    local job_name="evaluate-moe-${model}-xshield-amip-100km-to-3km-conus${SUFFIX}"

    # Only the teacher needs a beaker --dataset mount; the distilled bundle lives
    # on weka (already mounted below), so it takes no extra --dataset.
    local extra_mount=()
    [[ "$model" == "teacher" ]] && extra_mount=(--dataset "$DATASET_TEACHER:/checkpoints")

    gantry run \
        --name "$job_name" \
        --description "Evaluate MoE ${model} on CONUS holdout year 2023 (3 km)" \
        --workspace ai2/climate-titan \
        --priority urgent \
        --preemptible \
        --cluster ai2/titan \
        --beaker-image "$IMAGE" \
        --env WANDB_USERNAME="$BEAKER_USERNAME" \
        --env WANDB_NAME="$job_name" \
        --env WANDB_JOB_TYPE=inference \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        "${extra_mount[@]}" \
        --weka climate-default:/climate-default \
        --gpus $NGPU \
        --shared-memory 400GiB \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- torchrun --nproc_per_node $NGPU -m fme.downscaling.evaluator "$SCRIPT_PATH/${config}"
}

MODE="${1:-}"
[[ -z "$MODE" ]] && usage
shift

SUFFIX=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suffix)
            [[ -z "${2:-}" ]] && { echo "Error: --suffix requires a value"; usage; }
            SUFFIX="-$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

case "$MODE" in
    bundle)
        run_bundle ;;
    teacher)
        run_eval teacher config-teacher.yaml ;;
    distilled)
        run_eval distilled config-distilled.yaml ;;
    all)
        run_eval teacher config-teacher.yaml
        run_eval distilled config-distilled.yaml ;;
    *) echo "Unknown mode: $MODE"; usage ;;
esac
