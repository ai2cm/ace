#!/bin/bash

set -e
export GRPC_VERBOSITY=ERROR

# 1979-2014 rollout of one stage-2 checkpoint, scored exactly as ACE2.2-ERA5's four stage-2
# seeds were (long36-evaluator-config.yaml is ../2026-08-28-ace22-variants/long36-backfill/'s,
# unchanged, run at the same code, image and GPU type: 8 initial conditions, ~4 h on one B200).
#
#   bash run-long36.sh <label> <file in training_checkpoints/>
#   bash run-long36.sh ep01 best_ckpt.tar        # after: bash snapshot-ckpt.sh ep01 best_ckpt.tar
#   bash run-long36.sh final best_inference_ckpt.tar
#
# The checkpoint comes from dataset ace22-no-gmr-stage2-<label> if snapshot-ckpt.sh made one
# (needed while stage 2 is still running), otherwise from stage 2's committed result dataset.

LABEL=${1:?usage: run-long36.sh <label> <checkpoint file>}
CKPT_FILE=${2:?usage: run-long36.sh <label> <checkpoint file>}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STAGE2="ace22-no-gmr-stage2-rs0"
JOB_NAME="ace22-no-gmr-long36-$LABEL"
ACE_GIT_REF="394e41b5e07d1a3606d51faea1bd09a6fbddb492"
IMAGE="oliverwm/fme-deps-only-c82095065"

CONFIG_COMMIT=$(git -C "$SCRIPT_DIR" rev-parse HEAD)
if [[ -n "$(git -C "$SCRIPT_DIR" -c core.fileMode=false status --porcelain -- .)" ]]; then
    CONFIG_COMMIT="$CONFIG_COMMIT-dirty"
fi
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_IDENTITY="bhenn1983"  # differs from BEAKER_USERNAME; do not derive one from the other

if beaker experiment get "$BEAKER_USERNAME/$JOB_NAME" >/dev/null 2>&1; then
    echo "skip $JOB_NAME: experiment exists"
    exit 0
fi
SNAPSHOT="$BEAKER_USERNAME/ace22-no-gmr-stage2-$LABEL"
if beaker dataset get "$SNAPSHOT" >/dev/null 2>&1; then
    SNAPSHOT_ID=$(beaker dataset get "$SNAPSHOT" --format json | jq -r '.[0].id')
    MOUNT="$SNAPSHOT_ID:$CKPT_FILE:/ckpt.tar"
else
    STATUS=$(beaker experiment get "$BEAKER_USERNAME/$STAGE2" --format json \
        | python3 -c "import json, sys; j = json.load(sys.stdin)[0]['jobs'][-1]; print(j['status'].get('exitCode', ''), j['result']['beaker'])")
    read -r EXIT_CODE RESULT <<< "$STATUS"
    if [[ "$EXIT_CODE" != 0 ]]; then
        echo "ERROR: $STAGE2 has not exited 0; run snapshot-ckpt.sh $LABEL $CKPT_FILE first" >&2
        exit 1
    fi
    MOUNT="$RESULT:training_checkpoints/$CKPT_FILE:/ckpt.tar"
fi

CONFIG_B64=$(base64 < "$SCRIPT_DIR/long36-evaluator-config.yaml" | tr -d '\n')
echo "launch $JOB_NAME"
gantry run \
    --remote https://github.com/ai2cm/ace \
    --ref "$ACE_GIT_REF" \
    --name "$JOB_NAME" \
    --task-name "$JOB_NAME" \
    --description "ACE2.2-ERA5 without GMR: 1979-2014 rollout of stage-2 $CKPT_FILE ($LABEL)" \
    --beaker-image "$IMAGE" \
    --workspace ai2/ace \
    --priority high \
    --min-runtime 8h \
    --cluster ai2/titan \
    --env WANDB_USERNAME=$WANDB_IDENTITY \
    --env WANDB_NAME="$JOB_NAME" \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP=ace22-no-gmr \
    --env EXPERIMENT_CONFIG_COMMIT="$CONFIG_COMMIT" \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-${BEAKER_USERNAME} \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "$MOUNT" \
    --gpus 1 \
    --shared-memory 200GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- bash -c "echo '${CONFIG_B64}' | base64 -d > /tmp/config.yaml && torchrun --nproc_per_node 1 -m fme.ace.evaluator /tmp/config.yaml"
