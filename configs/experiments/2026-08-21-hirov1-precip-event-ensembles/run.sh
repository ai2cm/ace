#!/bin/bash
# Launch HiROv1 1000-member ensembles for the high 24h-accumulated-precip events
# found in the coarse (25km) central-US analysis. Five discrete events across three
# cases; each gets its own job and its own output zarr.
#
# Why --preemptible with --priority urgent: non-preemptible jobs in this workspace
# are capped at 2 GPUs, so --preemptible is required to get 4. Urgent priority means
# actual preemption effectively does not happen once a job has started. That matters
# because fme.downscaling.inference has no resume: ZarrWriter opens the store with
# mode "w-", so a restarted job fails on the existing zarr rather than continuing.
# Relaunching a case therefore requires removing its output directory from weka
# first.
#
# Why outputs go to weka rather than /results: each store is 7-9GB and beaker
# results are too slow to read from an interactive session, which is where the
# ensemble analysis happens.
#
# Usage: ./run.sh <case> [--suffix <suffix>]
#   case:     co-20210605 | wi-20210713 | scus-20210301 | scus-20220128
#             | scus-20220208 | all
#   --suffix: optional suffix appended to the job name

set -e

BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR#$REPO_ROOT/}"

cd $REPO_ROOT

NGPU=4
IMAGE="$(cat $REPO_ROOT/latest_deps_only_image.txt)"

# HiROv1 teacher checkpoints (from
# experiment/fastgen-distill 2026-07-28-hirov1-vs-spectral-conus-1gpu/run.sh).
DATASET_HIROV1=01KNM6H3JB1ZNS76HX17AAZRF7
SUBPATH_HIROV1=checkpoints

usage() {
    echo "Usage: $0 <case> [--suffix <suffix>]"
    echo "  case:     co-20210605 | wi-20210713 | scus-20210301 | scus-20220128"
    echo "            | scus-20220208 | all"
    echo "  --suffix: optional suffix appended to job name"
    exit 1
}

# Map short case name -> config basename (without the config- prefix or .yaml).
config_for_case() {
    case "$1" in
        co-20210605)   echo "co-training-storms-20210605" ;;
        wi-20210713)   echo "wi-training-storm-20210713" ;;
        scus-20210301) echo "scus-frontal-20210301" ;;
        scus-20220128) echo "scus-frontal-20220128" ;;
        scus-20220208) echo "scus-frontal-20220208" ;;
        *)             return 1 ;;
    esac
}

ALL_CASES="co-20210605 wi-20210713 scus-20210301 scus-20220128 scus-20220208"

run_case() {
    local case_name="$1"
    local config_name
    config_name=$(config_for_case "$case_name") || { echo "Unknown case: $case_name"; usage; }

    local job_name="hirov1-ensemble-1000-${config_name}${SUFFIX}"

    gantry run \
        --name "$job_name" \
        --description "HiROv1 1000-member 3km ensemble for high 24h-accum precip event ${config_name}" \
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
        --dataset "${DATASET_HIROV1}:${SUBPATH_HIROV1}:/checkpoints" \
        --weka climate-default:/climate-default \
        --gpus $NGPU \
        --shared-memory 100GiB \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- torchrun --nproc_per_node $NGPU -m fme.downscaling.inference \
            "$SCRIPT_PATH/config-${config_name}.yaml"
}

CASE="${1:-}"
[[ -z "$CASE" ]] && usage
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

if [[ "$CASE" == "all" ]]; then
    for c in $ALL_CASES; do
        run_case "$c"
    done
else
    run_case "$CASE"
fi
