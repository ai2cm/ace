#!/bin/bash
# Evaluate the two PRATE-only distilled students -- GAN-only baseline (f7z93y0a)
# vs. spectral-loss arm (i26sidsm) -- on the holdout year (2023) against X-SHiELD
# AMIP ground truth. Two regions per model: CONUS and the maritime continent.
#
# Both models share the same architecture/config; only the mounted checkpoint
# dataset differs, so the region configs (config-conus.yaml /
# config-maritime-continent.yaml) are reused across both models. Both are
# evaluated at student_checkpoints/best_student_tail.ckpt with
# num_diffusion_generation_steps: 2.
#
# Usage: ./run.sh <model> [--region <region>] [--suffix <suffix>]
#   model:    baseline | spectral | all
#   --region: conus (default) | maritime-continent | both
#   --suffix: optional suffix appended to the job name

set -e

BEAKER_USERNAME=$(conda run -n fme beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR#$REPO_ROOT/}"

cd $REPO_ROOT

NGPU=4
IMAGE="$(cat $REPO_ROOT/latest_deps_only_image.txt)"

# Training-run result datasets (from LOG.md registry). The student checkpoints
# live under fastgen/<experiment-name>/student_checkpoints inside each dataset.
DATASET_BASELINE=01KWX5CVSADN9KR44ZQN98QR1Y
SUBPATH_BASELINE=fastgen/ace-downscaling-distillation-fdistill-with-val-prate-baseline/student_checkpoints
DATASET_SPECTRAL=01KX00NA0DMZ99S3TKN1RYJKKQ
SUBPATH_SPECTRAL=fastgen/ace-downscaling-distillation-fdistill-with-val-prate-spectral-fix/student_checkpoints

usage() {
    echo "Usage: $0 <model> [--region <region>] [--suffix <suffix>]"
    echo "  model:    baseline | spectral | all"
    echo "  --region: conus (default) | maritime-continent | both"
    echo "  --suffix: optional suffix appended to job name"
    exit 1
}

run_eval() {
    local model="$1"        # baseline | spectral
    local region="$2"       # conus | maritime-continent
    local config="$3"
    local dataset_mount="$4"

    local job_name="evaluate-prate-${model}-xshield-amip-control-100km-to-3km-${region}${SUFFIX}"

    conda run -n fme gantry run \
        --name "$job_name" \
        --description "Evaluate PRATE ${model} student on ${region} holdout year 2023" \
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

# Dispatch one (model, region) pair.
run_pair() {
    local model="$1"
    local region="$2"
    local config dataset subpath

    case "$region" in
        conus)              config="config-conus.yaml" ;;
        maritime-continent) config="config-maritime-continent.yaml" ;;
        *) echo "Unknown region: $region"; usage ;;
    esac

    case "$model" in
        baseline) dataset="$DATASET_BASELINE"; subpath="$SUBPATH_BASELINE" ;;
        spectral) dataset="$DATASET_SPECTRAL"; subpath="$SUBPATH_SPECTRAL" ;;
        *) echo "Unknown model: $model"; usage ;;
    esac

    run_eval "$model" "$region" "$config" "${dataset}:${subpath}:/checkpoints"
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

case "$REGION" in
    conus|maritime-continent) REGIONS=("$REGION") ;;
    both) REGIONS=(conus maritime-continent) ;;
    *) echo "Unknown region: $REGION. Choose from: conus, maritime-continent, both"; usage ;;
esac

case "$MODEL" in
    baseline|spectral) MODELS=("$MODEL") ;;
    all) MODELS=(baseline spectral) ;;
    *) echo "Unknown model: $MODEL"; usage ;;
esac

for m in "${MODELS[@]}"; do
    for r in "${REGIONS[@]}"; do
        run_pair "$m" "$r"
    done
done
