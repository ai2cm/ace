#!/bin/bash

set -e
export GRPC_VERBOSITY=ERROR

# Ablations of ACE2.2's recipe at 4 degrees daily in AIMIP mode, one change each from the
# ACE2.2-like control except the last arm, 4 seeds per arm. Same code, image, data and +4 K pair
# as the ACE2.5 recipe's runs in ../2026-10-23-ace25-aimip-4deg/, the whole-recipe comparison:
#
#   ace22-like-4deg                          control: ACE2.2's recipe
#   ace22-like-4deg-no-gmr                   without global mean removal
#   ace22-like-4deg-diagnostic-near-surface  near-surface fields diagnosed, not stepped forward
#   ace22-like-4deg-no-bottleneck            without the spectral bottleneck
#   ace22-like-4deg-ace25-ft                 the ACE2.5 recipe's fine-tune, from the control's
#                                            pretrains
#   ace22-like-4deg-no-gmr-diagnostic-near-surface
#                                            without global mean removal and with near-surface
#                                            fields diagnosed (two changes)
#
# Each arm runs pretrain -> fine-tune -> 1979-2014 rollout and +4 K pair. Latent global-mean
# clipping acts only at inference, so the rollout and the pair are each run both as trained
# and with the clip off. The control and no-bottleneck arms also run a 1940-1978 rollout, clip
# on, on years outside training.
#
# Safe to rerun at any point. Each job is launched only if its beaker experiment does not
# exist yet and the stage it depends on has a job that exited 0; that job's result dataset
# is looked up here, so no ids are recorded in this file. Run it again after a stage
# finishes to launch the next.
#
#   bash run.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ACE25_DIR="$SCRIPT_DIR/../2026-10-23-ace25-aimip-4deg"
SEEDS="0 1 2 3"
declare -A PRETRAIN=(
    [ace22-like-4deg]=ace22-pretrain.yaml
    [ace22-like-4deg-no-gmr]=ace22-pretrain-no-gmr.yaml
    [ace22-like-4deg-diagnostic-near-surface]=ace22-pretrain-diagnostic-near-surface.yaml
    [ace22-like-4deg-no-bottleneck]=ace22-pretrain-no-bottleneck.yaml
    [ace22-like-4deg-no-gmr-diagnostic-near-surface]=ace22-pretrain-no-gmr-diagnostic-near-surface.yaml
)
ARMS="ace22-like-4deg ace22-like-4deg-no-gmr ace22-like-4deg-diagnostic-near-surface ace22-like-4deg-no-bottleneck ace22-like-4deg-ace25-ft ace22-like-4deg-no-gmr-diagnostic-near-surface"
# The fine-tune arm has no pretrain of its own and its own fine-tune config.
declare -A PRETRAIN_ARM=([ace22-like-4deg-ace25-ft]=ace22-like-4deg)
declare -A FINETUNE=([ace22-like-4deg-ace25-ft]=ace22-finetune-ace25.yaml)
# The evaluated model is the fine-tune's last EMA checkpoint: epoch 40 for ACE2.2's fine-tune,
# epoch 10 for ACE2.5's.
declare -A CKPT_EPOCH=([ace22-like-4deg-ace25-ft]=0010)

# Same code and image as the ACE2.5 recipe's runs.
ACE_GIT_REF="49ccc95df19c218c97064f41dcf4304597aaf894"
IMAGE="oliverwm/fme-deps-only-54045d546"

CONFIG_COMMIT=$(git -C "$SCRIPT_DIR" rev-parse HEAD)
if [[ -n "$(git -C "$SCRIPT_DIR" -c core.fileMode=false status --porcelain -- . "$ACE25_DIR")" ]]; then
    CONFIG_COMMIT="$CONFIG_COMMIT-dirty"
fi
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_IDENTITY="bhenn1983"  # differs from BEAKER_USERNAME; do not derive one from the other
CLIP_OFF_B64=$(base64 < "$SCRIPT_DIR/clip_off.py" | tr -d '\n')

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

# COMMAND runs after the config is written to /tmp/config.yaml and clip_off.py to /tmp.
launch () {
    local JOB_NAME=$1 CONFIG_PATH=$2 MOUNT=$3 JOB_TYPE=$4 COMMAND=$5
    local CONFIG_B64
    CONFIG_B64=$(base64 < "$CONFIG_PATH" | tr -d '\n')
    echo "launch $JOB_NAME ($JOB_TYPE)"
    gantry run \
        --remote https://github.com/ai2cm/ace \
        --ref "$ACE_GIT_REF" \
        --name "$JOB_NAME" \
        --task-name "$JOB_NAME" \
        --description "4-degree recipe ablations, AIMIP mode ($JOB_NAME)" \
        --beaker-image "$IMAGE" \
        --workspace ai2/ace \
        --priority high \
        --min-runtime 8h \
        --cluster ai2/jupiter-cirrascale-2 \
        --env WANDB_USERNAME=$WANDB_IDENTITY \
        --env WANDB_NAME="$JOB_NAME" \
        --env WANDB_JOB_TYPE="$JOB_TYPE" \
        --env WANDB_RUN_GROUP="4deg-recipe-ablations" \
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
        -- bash -c "echo '${CONFIG_B64}' | base64 -d > /tmp/config.yaml && echo '${CLIP_OFF_B64}' | base64 -d > /tmp/clip_off.py && $COMMAND"
}

# Launches JOB_NAME once DEPENDENCY (empty for none) has a successful job.
stage () {
    local JOB_NAME=$1 CONFIG_PATH=$2 DEPENDENCY=$3 MOUNT_PATH=$4 JOB_TYPE=$5 COMMAND=$6
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
    launch "$JOB_NAME" "$CONFIG_PATH" "$MOUNT" "$JOB_TYPE" "$COMMAND"
}

# The clip-off copy is made inside the job (/ckpt.tar is read-only).
CLIP_OFF="python /tmp/clip_off.py /ckpt.tar /tmp/ckpt.tar"
for ARM in $ARMS; do
    for SEED in $SEEDS; do
        PRE="${PRETRAIN_ARM[$ARM]:-$ARM}-pretrain-rs$SEED"
        FT="$ARM-finetune-rs$SEED"
        CKPT="training_checkpoints/ema_ckpt_${CKPT_EPOCH[$ARM]:-0040}.tar:/ckpt.tar"
        TRAIN="python -m fme.ace.train /tmp/config.yaml --override seed=$SEED"
        EVAL="python -m fme.ace.evaluator /tmp/config.yaml --override seed=$SEED"
        # fme.ace.inference for the +4 K pair: the evaluator at this ref cannot apply SST
        # perturbations.
        INFER="python -m fme.ace.inference /tmp/config.yaml --override seed=$SEED"
        if [[ -n "${PRETRAIN[$ARM]}" ]]; then
            stage "$PRE" "$SCRIPT_DIR/${PRETRAIN[$ARM]}" "" "" training "$TRAIN"
        fi
        stage "$FT" "$SCRIPT_DIR/${FINETUNE[$ARM]:-ace22-finetune.yaml}" "$PRE" "/weights" training "$TRAIN"
        # The fine-tune's inline long_36year rollout gives the trend with the clip on; these
        # repeat it through the evaluator so clip on and off compare directly.
        stage "$ARM-long36-rs$SEED" "$SCRIPT_DIR/long36-evaluator.yaml" \
            "$FT" "$CKPT" inference "$EVAL"
        stage "$ARM-long36-clipoff-rs$SEED" "$SCRIPT_DIR/long36-evaluator.yaml" \
            "$FT" "$CKPT" inference "$CLIP_OFF && $EVAL checkpoint_path=/tmp/ckpt.tar"
        for EXP in control p4k; do
            stage "$ARM-inference-$EXP-rs$SEED" "$ACE25_DIR/inference-$EXP.yaml" \
                "$FT" "$CKPT" inference "$INFER"
            stage "$ARM-inference-$EXP-clipoff-rs$SEED" "$ACE25_DIR/inference-$EXP.yaml" \
                "$FT" "$CKPT" inference "$CLIP_OFF && $INFER checkpoint_path=/tmp/ckpt.tar"
        done
        if [[ "$ARM" == ace22-like-4deg || "$ARM" == ace22-like-4deg-no-bottleneck ]]; then
            stage "$ARM-pre1979-rs$SEED" "$SCRIPT_DIR/pre1979-evaluator.yaml" \
                "$FT" "$CKPT" inference "$EVAL"
        fi
    done
done
