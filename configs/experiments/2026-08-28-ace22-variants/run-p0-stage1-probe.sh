#!/bin/bash

set -e
export GRPC_VERBOSITY=ERROR

# P0: probe the ACE2.2 stage-1 (pretrain) checkpoint with AIMIP-style inference, to test
# whether the secular warming gain and the E1 near-surface character are already present
# before the multi-step fine-tune. If they are, later variants can be compared after
# stage 1 alone (~180 GPU-h) instead of the full three-stage recipe (~400 GPU-h).
#
# Two jobs only: control and +4K, one realization each. The configs drop the 12 `gr`
# file entries for ta/hus/ua/va, which come from the stage-3 secondary decoder and do
# not exist in a stage-1 checkpoint; zg (h500) and TMP850 are core outputs and remain.
#
# --not-preemptible: these are ~75 min and inference is not resumable. The preemptible
# experiment on the 15-job sweep discarded 57% of its GPU-hours.

JOB_NAME_BASE="ace22-p0-stage1-probe"
JOB_GROUP="ace22-era5-6h-aimip"

# Stage-1 pretrain result dataset (wandb g94277n6, beaker 01KZYJ4HT4ZMZH296KBNWMPCQF).
STAGE1_RESULTS_DATASET="01KZYJ4HTBWED5VG3VFTRYKDRC"

# Same ace commit the checkpoint was trained at.
ACE_GIT_REF="fa856b459dc6c25b4d13b8e927d258aa7cefe543"
OUTPUT_ROOT="/climate-default/2026-08-28-ace22-p0-stage1-probe"
IC_PATH="/climate-default/2026-08-24-aimip-evaluation/aimip-evaluation-ics/1978-09-30_IC0.nc"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_IDENTITY="bhenn1983"  # differs from BEAKER_USERNAME; do not derive one from the other

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git rev-parse --show-toplevel)
CFG_DIR=$SCRIPT_DIR/p0-stage1-probe

launch_job () {
    local JOB_NAME=$1 TEMPLATE_CONFIG=$2 OVERRIDE=$3
    local CONFIG_B64
    CONFIG_B64=$(base64 < "$TEMPLATE_CONFIG" | tr -d '\n')

    gantry run \
        --remote https://github.com/ai2cm/ace \
        --ref $ACE_GIT_REF \
        --name $JOB_NAME \
        --task-name $JOB_NAME \
        --description 'P0 probe of the ACE2.2 stage-1 checkpoint' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        --not-preemptible \
        --cluster ai2/titan-cirrascale \
        --cluster ai2/jupiter-cirrascale-2 \
        --env WANDB_USERNAME=$WANDB_IDENTITY \
        --env WANDB_NAME=$JOB_NAME \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP=$JOB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-${BEAKER_USERNAME} \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $STAGE1_RESULTS_DATASET:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar \
        --gpus 1 \
        --shared-memory 50GiB \
        --weka climate-default:/climate-default \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- bash -c "echo '${CONFIG_B64}' | base64 -d > /tmp/p0-config.yaml && python -I -m fme.ace.inference /tmp/p0-config.yaml --override ${OVERRIDE}"
}

for EXPERIMENT in control p4k; do
    case $EXPERIMENT in
        control) CFG=$CFG_DIR/ace-p0-stage1-inference-config.yaml;     SUFFIX="" ;;
        p4k)     CFG=$CFG_DIR/ace-p0-stage1-inference-p4k-config.yaml; SUFFIX="-p4k" ;;
    esac
    JOB_NAME="${JOB_NAME_BASE}${SUFFIX}"
    OVERRIDE="initial_condition.path=${IC_PATH} experiment_dir=${OUTPUT_ROOT}/${JOB_NAME} seed=1"
    echo "Launching $JOB_NAME"
    launch_job "$JOB_NAME" "$CFG" "$OVERRIDE"
done
