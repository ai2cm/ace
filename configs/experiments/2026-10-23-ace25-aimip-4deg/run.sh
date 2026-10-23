#!/bin/bash

set -e
export GRPC_VERBOSITY=ERROR

# ACE2S paper recipe at 4 degrees in AIMIP mode: pretrain -> fine-tune -> +4 K evaluation.
#
# Safe to rerun at any point. Each job is launched only if its beaker experiment does not
# exist yet and the stage it depends on has a job that exited 0; that job's result dataset
# is looked up here, so no ids are recorded in this file. Run it again after a stage
# finishes to launch the next.
#
#   bash run.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NAME_PREFIX="ace25-aimip-4deg"
SEED=0

# Jeremy's 4-degree paper arm ran at f001ebb06 (pretrain) and efc6c39cc (fine-tune); fme/ is
# the same tree there and at 49ccc95, which holds the configs these are derived from.
ACE_GIT_REF="49ccc95df19c218c97064f41dcf4304597aaf894"
IMAGE="oliverwm/fme-deps-only-54045d546"

CONFIG_COMMIT=$(git -C "$SCRIPT_DIR" rev-parse HEAD)
if [[ -n "$(git -C "$SCRIPT_DIR" status --porcelain -- .)" ]]; then
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
    local COMMAND=$4
    local CONFIG_B64
    CONFIG_B64=$(base64 < "$CONFIG_PATH" | tr -d '\n')
    echo "launch $JOB_NAME"
    gantry run \
        --remote https://github.com/ai2cm/ace \
        --ref "$ACE_GIT_REF" \
        --name "$JOB_NAME" \
        --task-name "$JOB_NAME" \
        --description "ACE2S paper recipe, 4 degrees, AIMIP mode ($JOB_NAME)" \
        --beaker-image "$IMAGE" \
        --workspace ai2/ace \
        --priority high \
        --min-runtime 8h \
        --cluster ai2/jupiter-cirrascale-2 \
        --cluster ai2/titan-cirrascale \
        --env WANDB_USERNAME=$WANDB_IDENTITY \
        --env WANDB_NAME="$JOB_NAME" \
        --env WANDB_RUN_GROUP="$NAME_PREFIX" \
        --env EXPERIMENT_CONFIG_COMMIT="$CONFIG_COMMIT" \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-${BEAKER_USERNAME} \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        ${MOUNT:+--dataset "$MOUNT"} \
        --gpus 1 \
        --shared-memory 100GiB \
        --weka climate-default:/climate-default \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- bash -c "echo '${CONFIG_B64}' | base64 -d > /tmp/config.yaml && $COMMAND /tmp/config.yaml --override seed=${SEED}"
}

# Launches JOB_NAME once DEPENDENCY (empty for none) has a successful job.
stage () {
    local JOB_NAME=$1 CONFIG=$2 DEPENDENCY=$3 MOUNT_PATH=$4 COMMAND=$5
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
        MOUNT="$RESULT:$MOUNT_PATH"
    fi
    launch "$JOB_NAME" "$SCRIPT_DIR/$CONFIG" "$MOUNT" "$COMMAND"
}

PRETRAIN="$NAME_PREFIX-pretrain-rs$SEED"
FINETUNE="$NAME_PREFIX-finetune-rs$SEED"
stage "$PRETRAIN" pretrain.yaml "" "" "python -m fme.ace.train"
stage "$FINETUNE" finetune.yaml "$PRETRAIN" "/weights" "python -m fme.ace.train"
# The evaluated model is the fine-tune's epoch-10 EMA checkpoint, as in the paper.
for EXP in control p4k; do
    stage "$NAME_PREFIX-evaluate-$EXP-rs$SEED" "evaluate-$EXP.yaml" "$FINETUNE" \
        "training_checkpoints/ema_ckpt_0010.tar:/ckpt.tar" "python -m fme.ace.evaluator"
done
