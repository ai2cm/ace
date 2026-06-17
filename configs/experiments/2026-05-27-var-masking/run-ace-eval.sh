#!/bin/bash

set -e

CONFIG_FILENAME="${1:-ace-eval-config-4deg-AIMIP.yaml}"
JOB_NAME="${2:-ace-eval-config-4deg-AIMIP}"
JOB_GROUP="${3:-ace2-era5}"
EXISTING_RESULTS_DATASET="${4:-01KRF9EXM8CH80BVF8TQXFHM3J}"  # this contains the checkpoint to use for inference
CHECKPOINT_PATH="${5:-training_checkpoints/best_inference_ckpt.tar}"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
WANDB_PROJECT=${WANDB_PROJECT:-VarMasking}
BEAKER_WORKSPACE=${BEAKER_WORKSPACE:-ai2/climate-titan}
BEAKER_CLUSTER=${BEAKER_CLUSTER:-"ai2/jupiter"}
BEAKER_PRIORITY=${BEAKER_PRIORITY:-normal}
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

if [[ "${SKIP_VALIDATE:-0}" != "1" ]]; then
    python "$SCRIPT_PATH/run_eval_suite.py" --validate-only "$CONFIG_PATH"
fi

cluster_args=()
for cluster in $BEAKER_CLUSTER; do
    cluster_args+=(--cluster "$cluster")
done

cd $REPO_ROOT && gantry run \
    --name $JOB_NAME \
    --task-name $JOB_NAME \
    --description 'Run ACE2-ERA5 evaluator' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace "$BEAKER_WORKSPACE" \
    --priority "$BEAKER_PRIORITY" \
    --not-preemptible \
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
    -- python -I "$SCRIPT_PATH/run_eval_suite.py" $CONFIG_PATH
