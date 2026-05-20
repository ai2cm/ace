#!/bin/bash
# Evaluate distilled student checkpoints (DMD2 / f-distill) on the CONUS
# holdout year (2023) against X-SHiELD AMIP ground truth.
#
# Usage: ./run.sh <model> [--suffix <suffix>]
#   model:    dmd2 | fdistill | all
#   --suffix: optional suffix appended to the job name

set -e

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT

NGPU=8
IMAGE="$(cat $REPO_ROOT/latest_deps_only_image.txt)"
DATASET=01KRYPVQ3Z5YWQWND9X680GBMD

usage() {
    echo "Usage: $0 <model> [--suffix <suffix>]"
    echo "  model:    dmd2 | fdistill | all"
    echo "  --suffix: optional suffix appended to job name"
    exit 1
}

run_eval() {
    local model="$1"
    local config="$2"

    local job_name="evaluate-distilled-${model}-xshield-amip-control-100km-to-3km-conus${SUFFIX}"

    gantry run \
        --name "$job_name" \
        --description "Evaluate distilled ${model} student on CONUS holdout year 2023" \
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
        --dataset "$DATASET":checkpoints:/checkpoints \
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
            SUFFIX="-$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

case "$MODEL" in
    dmd2)     run_eval dmd2     config-dmd2.yaml ;;
    fdistill) run_eval fdistill config-fdistill.yaml ;;
    all)
        run_eval dmd2     config-dmd2.yaml
        run_eval fdistill config-fdistill.yaml
        ;;
    *)
        echo "Unknown model: $MODEL"
        usage
        ;;
esac
