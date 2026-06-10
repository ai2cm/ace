#!/bin/bash
# Evaluate distilled student checkpoints (DMD2 / f-distill / hirov1) on the
# holdout year (2023) against X-SHiELD AMIP ground truth.
#
# Usage: ./run.sh <model> [--region <region>] [--suffix <suffix>]
#   model:    dmd2 | fdistill | hirov1 | all
#   --region: conus (default) | maritime-continent
#             dmd2 only supports conus (no maritime-continent config)
#   --suffix: optional suffix appended to the job name

set -e

BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR#$REPO_ROOT/}"

cd $REPO_ROOT

NGPU=4
IMAGE="$(cat $REPO_ROOT/latest_deps_only_image.txt)"
DATASET_DMD2=01KRYPVQ3Z5YWQWND9X680GBMD
DATASET_FDISTILL=01KT25011Z8TZ08X1NFKD53RD1
DATASET_HIROV1=01KNM6H3JB1ZNS76HX17AAZRF7

usage() {
    echo "Usage: $0 <model> [--region <region>] [--suffix <suffix>]"
    echo "  model:    dmd2 | fdistill | fdistill-best-crps | hirov1 | all"
    echo "  --region: conus (default) | maritime-continent"
    echo "            dmd2 only supports conus"
    echo "  --suffix: optional suffix appended to job name"
    exit 1
}

run_eval() {
    local model="$1"
    local config="$2"
    local dataset_mount="$3"

    local job_name="evaluate-distilled-${model}-xshield-amip-control-100km-to-3km-${REGION}${SUFFIX}"

    gantry run \
        --name "$job_name" \
        --description "Evaluate distilled ${model} student on ${REGION} holdout year 2023" \
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

REGION="conus"
SUFFIX=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --region)
            [[ -z "${2:-}" ]] && { echo "Error: --region requires a value"; usage; }
            REGION="$2"
            shift 2
            ;;
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

# Resolve per-model config based on region.
case "$REGION" in
    conus)
        CONFIG_FDISTILL="config-fdistill.yaml"
        CONFIG_HIROV1="config-hirov1.yaml"
        ;;
    maritime-continent)
        CONFIG_FDISTILL="config-fdistill-maritime-continent.yaml"
        CONFIG_HIROV1="config-hirov1-maritime-continent.yaml"
        ;;
    *)
        echo "Unknown region: $REGION. Choose from: conus, maritime-continent"
        usage
        ;;
esac

case "$MODEL" in
    dmd2)
        [[ "$REGION" != "conus" ]] && { echo "Error: dmd2 only supports --region conus"; exit 1; }
        run_eval dmd2 config-dmd2.yaml "$DATASET_DMD2:/checkpoints"
        ;;
    fdistill)
        run_eval fdistill "$CONFIG_FDISTILL" "$DATASET_FDISTILL:fastgen/ace-downscaling-distillation-fdistill-with-val-intended-recipe/student_checkpoints:/checkpoints"
        ;;
    fdistill-best-crps)
        [[ "$REGION" != "conus" ]] && { echo "Error: fdistill-best-crps only supports --region conus"; exit 1; }
        run_eval fdistill-best-crps config-fdistill-best-crps.yaml "$DATASET_FDISTILL:fastgen/ace-downscaling-distillation-fdistill-with-val-intended-recipe/student_checkpoints:/checkpoints"
        ;;
    hirov1)
        run_eval hirov1 "$CONFIG_HIROV1" "$DATASET_HIROV1:checkpoints:/checkpoints"
        ;;
    all)
        [[ "$REGION" != "conus" ]] && { echo "Error: 'all' only supported with --region conus; run fdistill and hirov1 separately for other regions"; exit 1; }
        run_eval dmd2             config-dmd2.yaml             "$DATASET_DMD2:/checkpoints"
        run_eval fdistill         config-fdistill.yaml         "$DATASET_FDISTILL:fastgen/ace-downscaling-distillation-fdistill-with-val-intended-recipe/student_checkpoints:/checkpoints"
        run_eval fdistill-best-crps config-fdistill-best-crps.yaml "$DATASET_FDISTILL:fastgen/ace-downscaling-distillation-fdistill-with-val-intended-recipe/student_checkpoints:/checkpoints"
        run_eval hirov1           config-hirov1.yaml           "$DATASET_HIROV1:checkpoints:/checkpoints"
        ;;
    *)
        echo "Unknown model: $MODEL"
        usage
        ;;
esac
