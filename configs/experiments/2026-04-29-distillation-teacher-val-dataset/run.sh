#!/bin/bash
# Generate the pre-saved teacher validation dataset for distillation training.
#
# Default (single-variable teacher):
#   Output zarr: /climate-default/2026-04-29-distillation-teacher-val-dataset/conus_val_2023.zarr
#   Pass to fastgen_train.py via:
#     --val-dataset /climate-default/2026-04-29-distillation-teacher-val-dataset/conus_val_2023.zarr
#
# --multivar-moe (multivariate MoE teacher):
#   Output zarr: /climate-default/2026-06-09-distillation-teacher-moe-multivar-val-dataset/conus_multivar_val_2023.zarr
#   Pass to fastgen_train.py via:
#     --val-dataset /climate-default/2026-06-09-distillation-teacher-moe-multivar-val-dataset/conus_multivar_val_2023.zarr
#
# Both outputs have dims: (time, ensemble, latitude, longitude)
# Also pass --val-data-yaml <path/to/data-config.yaml> to fastgen_train.py.

set -e

MULTIVAR_MOE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --multivar-moe) MULTIVAR_MOE=true ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')
REPO_ROOT=$(git rev-parse --show-toplevel)
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')

cd $REPO_ROOT

NGPU=8
IMAGE="$(cat $REPO_ROOT/latest_distillation_image.txt)"

if $MULTIVAR_MOE; then
    TEACHER_DATASET=01KTCHVDHY0SATWH9E0AW2PDS6
    JOB_NAME=ace-downscaling-distillation-teacher-val-CONUS-2023-multivar-moe
    CONFIG_YAML=$SCRIPT_PATH/generate-teacher-val-multivar-moe.yaml
    DESCRIPTION="Generate pre-saved multivar MoE teacher validation dataset (CONUS 2023, n_ens=12)"
else
    TEACHER_DATASET=01KNM6H3JB1ZNS76HX17AAZRF7
    JOB_NAME=ace-downscaling-distillation-teacher-val-CONUS-2023
    CONFIG_YAML=$SCRIPT_PATH/generate-teacher-val.yaml
    DESCRIPTION="Generate pre-saved teacher validation dataset (CONUS 2023, n_ens=12)"
fi

gantry run \
    --name $JOB_NAME \
    --description "$DESCRIPTION" \
    --workspace ai2/climate-titan \
    --priority urgent \
    --cluster ai2/titan \
    --beaker-image $IMAGE \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_NAME=$JOB_NAME \
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
    -- torchrun --nproc-per-node $NGPU -m fme.downscaling.inference \
        $CONFIG_YAML \
        --overwrite
