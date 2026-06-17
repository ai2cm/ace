#!/bin/bash
# Test evaluate, predict, and inference entrypoints on a Europe domain (-16 to 30 lon)
# that straddles the prime meridian, validating the lon-0 crossing feature.
#
# Usage: ./run.sh <entrypoint>
#   entrypoint: evaluate | predict | inference | all

set -e

BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR#$REPO_ROOT/}"

cd $REPO_ROOT

NGPU=4
IMAGE="$(cat $REPO_ROOT/latest_deps_only_image.txt)"
DATASET_HIROV1=01KNM6H3JB1ZNS76HX17AAZRF7

usage() {
    echo "Usage: $0 <entrypoint>"
    echo "  entrypoint: evaluate | predict | inference | all"
    exit 1
}

run_gantry() {
    local entrypoint="$1"
    local config="$2"
    local module="$3"

    local job_name="lon-0-europe-testrun-${entrypoint}"

    gantry run \
        --name "$job_name" \
        --description "Test ${entrypoint} on prime-meridian-crossing Europe domain (lon -16 to 30)" \
        --workspace ai2/climate-titan \
        --priority normal \
        --preemptible \
        --cluster ai2/titan \
        --beaker-image "$IMAGE" \
        --env WANDB_USERNAME="$BEAKER_USERNAME" \
        --env WANDB_NAME="$job_name" \
        --env WANDB_JOB_TYPE=inference \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset "$DATASET_HIROV1:checkpoints:/checkpoints" \
        --weka climate-default:/climate-default \
        --gpus $NGPU \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- torchrun --nproc_per_node $NGPU -m $module "$SCRIPT_PATH/${config}"
}

ENTRYPOINT="${1:-}"
[[ -z "$ENTRYPOINT" ]] && usage

case "$ENTRYPOINT" in
    evaluate)
        run_gantry evaluate evaluate.yaml fme.downscaling.evaluator
        ;;
    predict)
        run_gantry predict predict.yaml fme.downscaling.predict
        ;;
    inference)
        run_gantry inference inference.yaml fme.downscaling.inference
        ;;
    all)
        run_gantry evaluate evaluate.yaml fme.downscaling.evaluator
        run_gantry predict predict.yaml fme.downscaling.predict
        run_gantry inference inference.yaml fme.downscaling.inference
        ;;
    *)
        echo "Unknown entrypoint: $ENTRYPOINT"
        usage
        ;;
esac
