#!/bin/bash

set -e

CONFIG_FILENAME="${1:-ace-inference-sst-config-4deg-p2k.yaml}"
JOB_NAME="${2:-ace-inference-sst-4deg}"
JOB_GROUP="${3:-ace2-var-masking-sst-perts-2026-07-08}"
EXISTING_RESULTS_DATASET="${4:-REPLACE_WITH_BEAKER_DATASET_ID}"  # contains the checkpoint to use for inference
CHECKPOINT_PATH="${5:-training_checkpoints/best_inference_ckpt.tar}"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/run_configs/$CONFIG_FILENAME  # generated configs live here
 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
WANDB_PROJECT=${WANDB_PROJECT:-VarMaskingC96}
BEAKER_WORKSPACE=${BEAKER_WORKSPACE:-ai2/climate-titan}
BEAKER_CLUSTER=${BEAKER_CLUSTER:-"ai2/titan"}
BEAKER_PRIORITY=${BEAKER_PRIORITY:-normal}
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

if [[ "${SKIP_VALIDATE:-0}" != "1" ]]; then
    python -m fme.ace.validate_config --config_type inference "$CONFIG_PATH"
fi

cluster_args=()
for cluster in $BEAKER_CLUSTER; do
    cluster_args+=(--cluster "$cluster")
done

cd $REPO_ROOT && gantry run \
    --name $JOB_NAME \
    --task-name $JOB_NAME \
    --description 'Run ACE2-ERA5 SST-perturbation inference' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace "$BEAKER_WORKSPACE" \
    --priority "$BEAKER_PRIORITY" \
    --preemptible \
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
    -- python -I -m fme.ace.inference $CONFIG_PATH
