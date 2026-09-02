#!/bin/bash

set -e

CONFIG_FILENAME="${1:-ace-eval-suite-config-4deg-AIMIP-nc-sfno-mask20-uniform-noco2-concat-seed0.yaml}"
JOB_NAME="${2:-ace2-var-mask-nc-sfno-mask20-uniform-noco2-concat-seed0-bestinf}"
JOB_GROUP="${3:-ace2-varmask-batch-comparison-eval-2026-08-26}"
EXISTING_RESULTS_DATASET="${4:?Beaker result dataset ID holding the checkpoint}"
CHECKPOINT_PATH="${5:-training_checkpoints/best_inference_ckpt.tar}"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
RUN_CONFIGS_SUBDIR=run_configs  # generated configs live here
CONFIG_PATH=$SCRIPT_PATH/$RUN_CONFIGS_SUBDIR/$CONFIG_FILENAME
 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
WANDB_PROJECT=${WANDB_PROJECT:-VarMaskingInfoComparison}
# ai2/ace is the workspace scripts/beaker_balancer manages, so CM_PRIORITY
# below is honoured here; it is ignored in a merely-observed workspace.
BEAKER_WORKSPACE=${BEAKER_WORKSPACE:-ai2/ace}
BEAKER_CLUSTER=${BEAKER_CLUSTER:-"ai2/titan ai2/jupiter"}
BEAKER_PRIORITY=${BEAKER_PRIORITY:-normal}
# Guaranteed runtime before Beaker may preempt the job; an evaluator preempted
# part-way through a suite has to redo every entry.
MIN_RUNTIME=${MIN_RUNTIME:-4h}
# Opts the job in to scripts/beaker_balancer, which keeps the team inside its
# urgent-priority allocation by moving opted-in jobs between priorities.
CM_PRIORITY=${CM_PRIORITY:-high}
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
    --min-runtime "$MIN_RUNTIME" \
    "${cluster_args[@]}" \
    --env CM_PRIORITY="$CM_PRIORITY" \
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
