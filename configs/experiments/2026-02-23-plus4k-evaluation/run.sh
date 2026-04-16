#!/bin/bash

set -e

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

NGPU=8
IMAGE="$(cat $REPO_ROOT/latest_deps_only_image.txt)"

usage() {
    echo "Usage: $0 <model> [experiment] [--suffix <suffix>]"
    echo "  model:      hirov1 | finetuned4k"
    echo "  experiment: control | plus4k | control-tropic | plus4k-tropic"
    echo "              (omit to run all experiments for the model)"
    echo "  --suffix:   optional suffix appended to job name (e.g. --suffix v2)"
    echo ""
    echo "  Note: finetuned4k also runs finetuned4k-loguniform control-tropic"
    echo "        when no experiment is specified"
    exit 1
}

run_experiment() {
    local dataset="$1"
    local model_suffix="$2"
    local config="$3"
    local climate_tag="$4"
    local region_tag="$5"

    local job_name="evaluate-HiRO${model_suffix}-xshield-amip-${climate_tag}-100km-to-3km-${region_tag}-generate${SUFFIX}"
    local config_path="$SCRIPT_PATH/${config}"
    local wandb_group=""

    gantry run \
        --name "$job_name" \
        --description 'Run 100km to 3km evaluation on coarsened X-SHiELD' \
        --workspace ai2/climate-titan \
        --priority urgent \
        --preemptible \
        --cluster ai2/titan \
        --beaker-image "$IMAGE" \
        --env WANDB_USERNAME="$BEAKER_USERNAME" \
        --env WANDB_NAME="$job_name" \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP="$wandb_group" \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset "$dataset":checkpoints:/checkpoints \
        --weka climate-default:/climate-default \
        --gpus $NGPU \
        --shared-memory 400GiB \
        --budget ai2/climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- torchrun --nproc_per_node $NGPU -m fme.downscaling.evaluator "$config_path"
}

run_all_experiments() {
    local dataset="$1"
    local model_suffix="$2"

    run_experiment "$dataset" "$model_suffix" config-control.yaml        control conus
    run_experiment "$dataset" "$model_suffix" config-plus4k.yaml         plus4K  conus
    run_experiment "$dataset" "$model_suffix" config-control-tropic.yaml control maritime
    run_experiment "$dataset" "$model_suffix" config-plus4k-tropic.yaml  plus4K  maritime
}

# Parse arguments
MODEL="${1:-}"
[[ -z "$MODEL" ]] && usage
shift

EXPERIMENT=""
SUFFIX=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suffix)
            [[ -z "${2:-}" ]] && { echo "Error: --suffix requires a value"; usage; }
            SUFFIX="-$2"
            shift 2
            ;;
        control|plus4k|control-tropic|plus4k-tropic)
            EXPERIMENT="$1"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

case "$MODEL" in
    hirov1)
        DATASET=01K8RWE83W8BEEAT2KRS94FVCD
        MODEL_SUFFIX=""
        ;;
    finetuned4k)
        DATASET=01KNM6H3JB1ZNS76HX17AAZRF7
        MODEL_SUFFIX="-finetuned4k"
        ;;
    *)
        echo "Unknown model: $MODEL"
        usage
        ;;
esac

if [[ -n "$EXPERIMENT" ]]; then
    case "$EXPERIMENT" in
        control)       run_experiment "$DATASET" "$MODEL_SUFFIX" config-control.yaml        control conus     ;;
        plus4k)        run_experiment "$DATASET" "$MODEL_SUFFIX" config-plus4k.yaml         plus4K  conus     ;;
        control-tropic)run_experiment "$DATASET" "$MODEL_SUFFIX" config-control-tropic.yaml control maritime  ;;
        plus4k-tropic) run_experiment "$DATASET" "$MODEL_SUFFIX" config-plus4k-tropic.yaml  plus4K  maritime  ;;
    esac
else
    run_all_experiments "$DATASET" "$MODEL_SUFFIX"
    if [[ "$MODEL" == "finetuned4k" ]]; then
        run_experiment 01KNN2VFZC9AAK4NQR7QK5REDC "-finetuned4k-loguniform" \
            config-control-tropic.yaml control maritime
    fi
fi
