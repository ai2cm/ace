#!/bin/bash

set -e
export GRPC_VERBOSITY=ERROR

# ACE2.2-ERA5 without global mean removal: stage 1 (1-step pretrain) -> stage 2 (multi-step
# fine-tune from stage 1's best_ckpt.tar), one seed.
#
# Safe to rerun at any point. Each job is launched only if its beaker experiment does not
# exist yet and the stage it depends on has a job that exited 0; that job's result dataset
# is looked up here, so no ids are recorded in this file. Run it again after a stage
# finishes to launch the next.
#
#   bash run.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NAME_PREFIX="ace22-no-gmr"
SEED=0

# The code and image ACE2.2-ERA5 was trained with (ai2cm/ACE2.1-ERA5-AIMIP
# scripts/run-ace2.2-6h-train.sh and configs/ace2.2-era5/deps_only_image.txt).
ACE_GIT_REF="394e41b5e07d1a3606d51faea1bd09a6fbddb492"
IMAGE="oliverwm/fme-deps-only-c82095065"
# As for ACE2.2-ERA5 seeds 1-3. batch_size is global, so the GPU count does not change the
# experiment.
N_GPUS=8

CONFIG_COMMIT=$(git -C "$SCRIPT_DIR" rev-parse HEAD)
if [[ -n "$(git -C "$SCRIPT_DIR" -c core.fileMode=false status --porcelain -- .)" ]]; then
    CONFIG_COMMIT="$CONFIG_COMMIT-dirty"
fi
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_IDENTITY="bhenn1983"  # differs from BEAKER_USERNAME; do not derive one from the other

experiment_exists () {
    beaker experiment get "$BEAKER_USERNAME/$1" >/dev/null 2>&1
}

# Result dataset of the experiment's job that exited 0, or nothing (also when the
# experiment does not exist yet).
succeeded_result () {
    { beaker experiment get "$BEAKER_USERNAME/$1" --format json 2>/dev/null || true; } | python3 -c "
import json, sys
text = sys.stdin.read().strip()
jobs = json.loads(text)[0]['jobs'] if text else []
ok = [j for j in jobs if (j.get('status') or {}).get('exitCode') == 0]
print((ok[-1].get('result') or {}).get('beaker', '') if ok else '')"
}

launch () {
    local JOB_NAME=$1
    local CONFIG_PATH=$2
    local MOUNT=$3
    local CONFIG_B64
    CONFIG_B64=$(base64 < "$CONFIG_PATH" | tr -d '\n')
    echo "launch $JOB_NAME"
    gantry run \
        --remote https://github.com/ai2cm/ace \
        --ref "$ACE_GIT_REF" \
        --name "$JOB_NAME" \
        --task-name "$JOB_NAME" \
        --description "ACE2.2-ERA5 without global mean removal ($JOB_NAME)" \
        --beaker-image "$IMAGE" \
        --workspace ai2/ace \
        --priority high \
        --min-runtime 8h \
        --cluster ai2/jupiter \
        --env WANDB_USERNAME=$WANDB_IDENTITY \
        --env WANDB_NAME="$JOB_NAME" \
        --env WANDB_JOB_TYPE=training \
        --env WANDB_RUN_GROUP="$NAME_PREFIX" \
        --env EXPERIMENT_CONFIG_COMMIT="$CONFIG_COMMIT" \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-${BEAKER_USERNAME} \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        ${MOUNT:+--dataset "$MOUNT"} \
        --gpus $N_GPUS \
        --shared-memory 400GiB \
        --weka climate-default:/climate-default \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- bash -c "echo '${CONFIG_B64}' | base64 -d > /tmp/config.yaml && torchrun --nproc_per_node $N_GPUS -m fme.ace.train /tmp/config.yaml --override seed=${SEED}"
}

# Launches JOB_NAME once DEPENDENCY (empty for none) has a successful job.
stage () {
    local JOB_NAME=$1 CONFIG=$2 DEPENDENCY=$3
    if experiment_exists "$JOB_NAME"; then
        echo "skip $JOB_NAME: experiment exists"
        return
    fi
    local MOUNT=""
    if [[ -n "$DEPENDENCY" ]]; then
        local RESULT
        RESULT=$(succeeded_result "$DEPENDENCY")
        if [[ -z "$RESULT" ]]; then
            echo "wait $JOB_NAME: $DEPENDENCY has no job that exited 0 yet"
            return
        fi
        MOUNT="$RESULT:/weights"
    fi
    launch "$JOB_NAME" "$SCRIPT_DIR/$CONFIG" "$MOUNT"
}

STAGE1="$NAME_PREFIX-stage1-rs$SEED"
stage "$STAGE1" train.yaml ""
stage "$NAME_PREFIX-stage2-rs$SEED" multi-step-ft.yaml "$STAGE1"
