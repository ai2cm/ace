#!/bin/bash
# Convert FastGen distillation checkpoints to ACE format and save as a
# Beaker dataset.  Outputs land in /results and are automatically captured.
#
# Outputs (in resulting Beaker dataset):
#   dmd2_student.ckpt    — DMD2 student, 4-step inference
#   fdistill_student.ckpt — f-distill (forward-KL) student, 4-step inference

set -e

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT

IMAGE="$(cat $REPO_ROOT/latest_distillation_image.txt)"
JOB_NAME=ace-downscaling-distillation-student-checkpoints

TEACHER_DATASET=01KNM6H3JB1ZNS76HX17AAZRF7
DMD2_DATASET=01KQAY19MH9G9XJDTWYRAV6Z3R
FDISTILL_DATASET=01KQAQSCPS35HT30G5C2GNHP2S

gantry run \
    --name $JOB_NAME \
    --description "Convert DMD2 and f-distill FastGen checkpoints to ACE format" \
    --workspace ai2/climate-titan \
    --priority urgent \
    --cluster ai2/titan \
    --beaker-image $IMAGE \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset $TEACHER_DATASET:checkpoints:/checkpoints \
    --dataset $DMD2_DATASET:/dmd2 \
    --dataset $FDISTILL_DATASET:/fdistill \
    --weka climate-default:/climate-default \
    --gpus 1 \
    --shared-memory 10GiB \
    --budget ai2/climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- python $SCRIPT_PATH/convert.py
