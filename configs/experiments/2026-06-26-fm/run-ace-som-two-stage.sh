#!/bin/bash

# Two-stage slab-ocean equilibrium inference (paper's
# run-ace-equilibrium-climate-inference.sh): a spin-up inference writing
# /results/spin-up/restart.nc, then the main inference initialized from it.

set -e

SPIN_UP_CONFIG_FILENAME="${1:-run_configs/ace-som-eq-spinup-config-4deg-1xCO2-ic1.yaml}"
MAIN_CONFIG_FILENAME="${2:-run_configs/ace-som-eq-main-config-4deg-1xCO2-ic1.yaml}"
JOB_NAME="${3:-ace-som-eq}"
JOB_GROUP="${4:-ace2-fm-som-2026-06-26}"
EXISTING_RESULTS_DATASET="${5:-REPLACE_WITH_BEAKER_DATASET_ID}"  # contains the checkpoint to use for inference
CHECKPOINT_PATH="${6:-training_checkpoints/best_inference_ckpt.tar}"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
SPIN_UP_CONFIG_PATH=$SCRIPT_PATH/$SPIN_UP_CONFIG_FILENAME
MAIN_CONFIG_PATH=$SCRIPT_PATH/$MAIN_CONFIG_FILENAME
 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
WANDB_PROJECT=${WANDB_PROJECT:-FM}
BEAKER_WORKSPACE=${BEAKER_WORKSPACE:-ai2/climate-titan}
BEAKER_CLUSTER=${BEAKER_CLUSTER:-"ai2/titan"}
BEAKER_PRIORITY=${BEAKER_PRIORITY:-normal}
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

if [[ "${SKIP_VALIDATE:-0}" != "1" ]]; then
    python -m fme.ace.validate_config --config_type inference "$SPIN_UP_CONFIG_PATH"
    python -m fme.ace.validate_config --config_type inference "$MAIN_CONFIG_PATH"
fi

cluster_args=()
for cluster in $BEAKER_CLUSTER; do
    cluster_args+=(--cluster "$cluster")
done

cd $REPO_ROOT && gantry run \
    --name $JOB_NAME \
    --task-name $JOB_NAME \
    --description 'Run ACE FM slab-ocean equilibrium inference (spin-up then main)' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace "$BEAKER_WORKSPACE" \
    --priority "$BEAKER_PRIORITY" \
    "${cluster_args[@]}" \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$JOB_NAME" \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP="$JOB_GROUP" \
    --env WANDB_PROJECT="$WANDB_PROJECT" \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset $EXISTING_RESULTS_DATASET:$CHECKPOINT_PATH:/ckpt.tar \
    --gpus 1 \
    --shared-memory 50GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    --allow-dirty \
    -- bash -c "python -I -m fme.ace.inference $SPIN_UP_CONFIG_PATH && python -I -m fme.ace.inference $MAIN_CONFIG_PATH"
