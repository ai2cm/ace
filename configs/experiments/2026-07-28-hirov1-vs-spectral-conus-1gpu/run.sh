#!/bin/bash
# Single-GPU CONUS eval: hirov1 baseline (full diffusion) vs the spectral-loss
# distilled student (i26sidsm, 2-step fastgen), holdout year 2023 against
# X-SHiELD AMIP ground truth.
#
# Why one GPU: fme/core/histogram.py's ComparedDynamicTailsHistograms performs no
# cross-rank reduction, so on a multi-rank eval the logged
# histogram/prediction_frac_of_target/* tail ratios come from a single rank's
# shard of the data. That shard is contiguous in time, not a random subsample:
# the evaluator builds with train=False, so PairedDataLoaderConfig._get_sampler
# returns a ContiguousDistributedSampler and rank 0 gets indices[0:N/nranks] --
# on 4 ranks over CONUS 2023, roughly Jan through early Apr, with no summer
# convection. Measured cost: ground-truth percentiles understated 8% (99.9999th)
# and 21% (99.99th). CRPS / RMSE / power_spectrum are unaffected -- Mean and
# MeanComparison reduce via TensorDictAccumulator.get_distributed_mean() -- so
# those metrics remain comparable to the earlier 4-GPU runs (flzvb6tp/x2nyzmzh).
# Only the tails change meaning here. Fixing the reduction properly is a durable
# pipeline change (needs a spec; would also shift every training-time
# best_*_tail.ckpt selector), tracked as a follow-up in the distillation LOG.
#
# Not --preemptible: one GPU makes this ~4x longer than the 4-GPU runs and the
# evaluator is a single pass with no resume, so a preemption restarts from zero.
#
# Usage: ./run.sh <model> [--suffix <suffix>]
#   model:  hirov1 | spectral | all
#   --suffix: optional suffix appended to the job name

set -e

BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR#$REPO_ROOT/}"

cd $REPO_ROOT

NGPU=1
IMAGE="$(cat $REPO_ROOT/latest_deps_only_image.txt)"

# hirov1 checkpoints (from 2026-05-20-distilled-model-eval/run.sh).
DATASET_HIROV1=01KNM6H3JB1ZNS76HX17AAZRF7
SUBPATH_HIROV1=checkpoints
# i26sidsm training-run results (from the distillation LOG.md registry).
DATASET_SPECTRAL=01KX00NA0DMZ99S3TKN1RYJKKQ
SUBPATH_SPECTRAL=fastgen/ace-downscaling-distillation-fdistill-with-val-prate-spectral-fix/student_checkpoints

usage() {
    echo "Usage: $0 <model> [--suffix <suffix>]"
    echo "  model:    hirov1 | spectral | all"
    echo "  --suffix: optional suffix appended to job name"
    exit 1
}

run_eval() {
    local model="$1"
    local config="$2"
    local dataset_mount="$3"

    local job_name="evaluate-${model}-xshield-amip-control-100km-to-3km-conus-1gpu${SUFFIX}"

    gantry run \
        --name "$job_name" \
        --description "Single-GPU eval of ${model} on CONUS holdout year 2023 (whole-dataset tail histograms)" \
        --workspace ai2/climate-titan \
        --priority urgent \
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
        --shared-memory 100GiB \
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

run_hirov1() {
    run_eval hirov1 config-hirov1.yaml "${DATASET_HIROV1}:${SUBPATH_HIROV1}:/checkpoints"
}

run_spectral() {
    run_eval spectral config-spectral.yaml "${DATASET_SPECTRAL}:${SUBPATH_SPECTRAL}:/checkpoints"
}

case "$MODEL" in
    hirov1)   run_hirov1 ;;
    spectral) run_spectral ;;
    all)      run_hirov1; run_spectral ;;
    *)        echo "Unknown model: $MODEL"; usage ;;
esac
