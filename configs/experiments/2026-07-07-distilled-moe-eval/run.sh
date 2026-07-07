#!/bin/bash
# Evaluate the distilled 2-step MoE student vs the original bundled MoE teacher
# on the CONUS 2023 holdout year, against 3 km X-SHiELD AMIP ground truth.
# Both configs share identical data/patch/n_samples, so the resulting wandb
# metrics (CRPS / spectra / tails, project andrep-downscaling) are directly
# comparable. Runs stats that don't otherwise exist for the bundled teacher.
#
# Usage: ./run.sh <model> [--suffix <suffix>]
#   model:  teacher | distilled | all
#   --suffix: optional suffix appended to the job name
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
# TODO: beaker dataset holding the assembled distilled student bundle
# (distilled_moe_bundle.ckpt), produced by DenoisingMoEStudentConfig(...).save().
DATASET_DISTILLED=REPLACE_WITH_DISTILLED_BUNDLE_DATASET_ID

usage() {
    echo "Usage: $0 <teacher|distilled|all> [--suffix <suffix>]"
    exit 1
}

run_eval() {
    local model="$1"
    local config="$2"
    local dataset_mount="$3"
    local job_name="evaluate-moe-${model}-xshield-amip-100km-to-3km-conus${SUFFIX}"

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
        --dataset "$dataset_mount" \
        --weka climate-default:/climate-default \
        --gpus $NGPU \
        --shared-memory 400GiB \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- torchrun --nproc_per_node $NGPU -m fme.downscaling.evaluator "$SCRIPT_PATH/${config}"
}

MODEL="${1:-}"
[[ -z "$MODEL" ]] && usage
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

case "$MODEL" in
    teacher)
        run_eval teacher config-teacher.yaml "$DATASET_TEACHER:/checkpoints" ;;
    distilled)
        run_eval distilled config-distilled.yaml "$DATASET_DISTILLED:/checkpoints" ;;
    all)
        run_eval teacher   config-teacher.yaml   "$DATASET_TEACHER:/checkpoints"
        run_eval distilled config-distilled.yaml "$DATASET_DISTILLED:/checkpoints" ;;
    *) echo "Unknown model: $MODEL"; usage ;;
esac
