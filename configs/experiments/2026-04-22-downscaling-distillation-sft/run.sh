#!/bin/bash
# Usage: ./run.sh <method>
#   method: sft | dmd2 | fdistill | scm

set -e

METHOD="${1:?Usage: $0 <sft|dmd2|fdistill|scm>}"

case "$METHOD" in
    sft)
        CONFIG="fme/downscaling/distillation/configs/sft_spike.py"
        DESCRIPTION="ACE downscaling SFT distillation spike"
        ;;
    dmd2)
        CONFIG="fme/downscaling/distillation/configs/dmd2_baseline_spike.py"
        DESCRIPTION="ACE downscaling DMD2 distillation spike"
        ;;
    fdistill)
        CONFIG="fme/downscaling/distillation/configs/fdistill_kl_spike.py"
        DESCRIPTION="ACE downscaling f-distill (forward-KL) spike"
        ;;
    scm)
        CONFIG="fme/downscaling/distillation/configs/scm_spike.py"
        DESCRIPTION="ACE downscaling sCM distillation spike"
        ;;
    *)
        echo "Unknown method: $METHOD. Choose from: sft, dmd2, fdistill, scm" >&2
        exit 1
        ;;
esac

JOB_NAME="ace-downscaling-distillation-${METHOD}"

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT

NGPU=4
IMAGE="$(cat $REPO_ROOT/latest_distillation_image.txt)"

TEACHER_DATASET=01KNM6H3JB1ZNS76HX17AAZRF7
ACE_TEACHER_CKPT=/checkpoints/best_histogram_tail.ckpt

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
    --budget ai2/climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc-per-node $NGPU -m fme.downscaling.distillation.fastgen_train \
        --config $CONFIG \
        --teacher-checkpoint $ACE_TEACHER_CKPT \
        --teacher-num-steps 15 \
        --data-yaml $SCRIPT_PATH/data-config.yaml \
        - log_config.name=$JOB_NAME
