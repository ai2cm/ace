#!/bin/bash
# Distillation training with BestStudentCheckpointCallback enabled.
#
# Saves best_student.ckpt (ACE format, by validation CRPS) into /results so
# it is captured as a Beaker dataset artifact alongside the raw .pth files.
#
# Usage: ./run.sh <method> [--suffix <variant>] [--moe-teacher]
#   method:      dmd2 | fdistill | scm
#   --suffix:    optional training-variant tag appended to JOB_NAME, e.g.
#                "1step" → ace-downscaling-distillation-fdistill-with-val-1step.
#                Useful when running multiple variants in parallel so each gets
#                a distinct wandb run name.
#   --moe-teacher: use the bundled multivariate MoE checkpoint
#                (01KTCHVDHY0SATWH9E0AW2PDS6 / bundled_moe_multivariate.ckpt)
#                as the teacher instead of the default single-model checkpoint.
#                Note: DMD2 + MoE teacher requires additional discriminator
#                wiring via the primary expert; fdistill/scm are fully supported.

set -e

METHOD="${1:?Usage: $0 <dmd2|fdistill|scm> [--suffix <variant>] [--moe-teacher]}"
shift

SUFFIX=""
MOE_TEACHER=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suffix)
            SUFFIX="${2:?--suffix requires a value}"
            shift 2
            ;;
        --moe-teacher)
            MOE_TEACHER=true
            shift
            ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 1
            ;;
    esac
done

case "$METHOD" in
    dmd2)
        CONFIG="fme/downscaling/distillation/configs/dmd2_baseline_spike.py"
        DESCRIPTION="ACE downscaling DMD2 distillation with val CRPS checkpoint"
        ;;
    fdistill)
        CONFIG="fme/downscaling/distillation/configs/fdistill_kl_spike.py"
        DESCRIPTION="ACE downscaling f-distill (forward-KL) with val CRPS checkpoint"
        ;;
    scm)
        CONFIG="fme/downscaling/distillation/configs/scm_spike.py"
        DESCRIPTION="ACE downscaling sCM distillation with val CRPS checkpoint"
        ;;
    *)
        echo "Unknown method: $METHOD. Choose from: dmd2, fdistill, scm" >&2
        exit 1
        ;;
esac

JOB_NAME="ace-downscaling-distillation-${METHOD}-with-val${SUFFIX:+-${SUFFIX}}"

BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
# Relative path from repo root to this script's directory — stable regardless
# of where the script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR#$REPO_ROOT/}"

cd $REPO_ROOT

NGPU=4
IMAGE="$(cat $REPO_ROOT/latest_distillation_image.txt)"

if [[ "$MOE_TEACHER" == "true" ]]; then
    # Bundled multivariate MoE teacher: precip + winds + pressure.
    TEACHER_DATASET=01KTCHVDHY0SATWH9E0AW2PDS6
    TEACHER_CKPT_FLAG="--teacher-moe-checkpoint /checkpoints/bundled_moe_multivariate.ckpt"
    JOB_NAME="${JOB_NAME}-moe-teacher"
    VAL_DATASET=/climate-default/2026-06-09-distillation-teacher-moe-multivar-val-dataset/conus_multivar_val_2023.zarr
else
    # Default single-model teacher.
    TEACHER_DATASET=01KNM6H3JB1ZNS76HX17AAZRF7:checkpoints
    TEACHER_CKPT_FLAG="--teacher-checkpoint /checkpoints/best_histogram_tail.ckpt"
    VAL_DATASET=/climate-default/2026-04-29-distillation-teacher-val-dataset/conus_val_2023.zarr
fi

gantry run \
    --name $JOB_NAME \
    --description "$DESCRIPTION" \
    --workspace ai2/climate-titan \
    --priority urgent \
    --preemptible \
    --cluster ai2/titan \
    --beaker-image $IMAGE \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env WANDB_JOB_TYPE=distillation \
    --env FASTGEN_OUTPUT_ROOT=/results \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset $TEACHER_DATASET:/checkpoints \
    --weka climate-default:/climate-default \
    --gpus $NGPU \
    --shared-memory 100GiB \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc-per-node $NGPU -m fme.downscaling.distillation.fastgen_train \
        --config $CONFIG \
        $TEACHER_CKPT_FLAG \
        --teacher-num-steps 15 \
        --data-yaml $SCRIPT_PATH/data-config.yaml \
        --val-dataset $VAL_DATASET \
        --val-data-yaml $SCRIPT_PATH/val-data-config.yaml \
        - log_config.name=$JOB_NAME
