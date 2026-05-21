#!/bin/bash
# Distillation training with BestStudentCheckpointCallback enabled.
#
# Saves best_student.ckpt (ACE format, by validation CRPS) into /results so
# it is captured as a Beaker dataset artifact alongside the raw .pth files.
#
# Usage: ./run.sh <method>
#   method: dmd2 | fdistill

set -e

METHOD="${1:?Usage: $0 <dmd2|fdistill>}"

case "$METHOD" in
    dmd2)
        CONFIG="fme/downscaling/distillation/configs/dmd2_baseline_spike.py"
        DESCRIPTION="ACE downscaling DMD2 distillation with val CRPS checkpoint"
        ;;
    fdistill)
        CONFIG="fme/downscaling/distillation/configs/fdistill_kl_spike.py"
        DESCRIPTION="ACE downscaling f-distill (forward-KL) with val CRPS checkpoint"
        ;;
    *)
        echo "Unknown method: $METHOD. Choose from: dmd2, fdistill" >&2
        exit 1
        ;;
esac

JOB_NAME="ace-downscaling-distillation-${METHOD}-with-val"

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT

NGPU=8
IMAGE="$(cat $REPO_ROOT/latest_distillation_image.txt)"

TEACHER_DATASET=01KNM6H3JB1ZNS76HX17AAZRF7
ACE_TEACHER_CKPT=/checkpoints/best_histogram_tail.ckpt
VAL_DATASET=/climate-default/2026-04-29-distillation-teacher-val-dataset/conus_val_2023.zarr

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
    --env ACE_TEACHER_CKPT=$ACE_TEACHER_CKPT \
    --env FASTGEN_OUTPUT_ROOT=/results \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset $TEACHER_DATASET:checkpoints:/checkpoints \
    --weka climate-default:/climate-default \
    --gpus $NGPU \
    --shared-memory 100GiB \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc-per-node $NGPU -m fme.downscaling.distillation.fastgen_train \
        --config $CONFIG \
        --teacher-checkpoint $ACE_TEACHER_CKPT \
        --teacher-num-steps 15 \
        --data-yaml $SCRIPT_PATH/data-config.yaml \
        --val-dataset $VAL_DATASET \
        --val-data-yaml $SCRIPT_PATH/val-data-config.yaml \
        - log_config.name=$JOB_NAME
