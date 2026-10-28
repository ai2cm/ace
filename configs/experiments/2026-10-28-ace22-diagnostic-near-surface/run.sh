#!/bin/bash

set -e
export GRPC_VERBOSITY=ERROR

# ACE2.2-ERA5 with diagnostic near-surface fields, with and without global mean removal, seeds
# 0-3 of each:
#
#   ace22-diagnostic-near-surface         train.yaml
#   ace22-no-gmr-diagnostic-near-surface  train-no-gmr.yaml
#
# Each seed runs stage 1 (1-step pretrain) -> stage 2 (multi-step fine-tune from stage 1's
# best_ckpt.tar) -> the 1979-2014 rollout and the control / +4 K SST pair on stage 2's
# best_inference_ckpt.tar, each as trained and with the latent clip off.
#
# Safe to rerun at any point. Each job is launched only if its beaker experiment does not
# exist yet and the stage it depends on has a job that exited 0; that job's result dataset
# is looked up here, so no ids are recorded in this file. Run it again after a stage
# finishes to launch the next.
#
#   bash run.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SEEDS="0 1 2 3"
declare -A STAGE1_CONFIG=(
    [ace22-diagnostic-near-surface]=train.yaml
    [ace22-no-gmr-diagnostic-near-surface]=train-no-gmr.yaml
)
ARMS="ace22-no-gmr-diagnostic-near-surface ace22-diagnostic-near-surface"

# The code and image ACE2.2-ERA5 was trained with (ai2cm/ACE2.1-ERA5-AIMIP
# scripts/run-ace2.2-6h-train.sh and configs/ace2.2-era5/deps_only_image.txt).
ACE_GIT_REF="394e41b5e07d1a3606d51faea1bd09a6fbddb492"
IMAGE="oliverwm/fme-deps-only-c82095065"
# As for ACE2.2-ERA5 seeds 1-3. batch_size is global, so the GPU count does not change the
# experiment.
TRAIN_GPUS=8
TRAIN_CLUSTERS="ai2/jupiter"
EVAL_CLUSTERS="ai2/titan ai2/jupiter"
# The +4 K pair's inputs and outputs, as for ACE2.2-ERA5's seeds.
IC_PATH="/climate-default/2026-08-24-aimip-evaluation/aimip-evaluation-ics/1978-09-30_IC0.nc"
OUTPUT_ROOT="/climate-default/2026-10-28-ace22-diagnostic-near-surface-perturbation"
CKPT="training_checkpoints/best_inference_ckpt.tar"

CONFIG_COMMIT=$(git -C "$SCRIPT_DIR" rev-parse HEAD)
if [[ -n "$(git -C "$SCRIPT_DIR" -c core.fileMode=false status --porcelain -- .)" ]]; then
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
    local JOB_NAME=$1 CONFIG_PATH=$2 MOUNT=$3 ARM=$4 JOB_TYPE=$5 COMMAND=$6
    local CONFIG_B64 GPUS SHM CLUSTERS CLUSTER_ARGS=()
    CONFIG_B64=$(base64 < "$CONFIG_PATH" | tr -d '\n')
    if [[ $JOB_TYPE == training ]]; then
        GPUS=$TRAIN_GPUS SHM=400GiB CLUSTERS=$TRAIN_CLUSTERS
    else
        GPUS=1 SHM=200GiB CLUSTERS=$EVAL_CLUSTERS
    fi
    for CLUSTER in $CLUSTERS; do
        CLUSTER_ARGS+=(--cluster "$CLUSTER")
    done
    echo "launch $JOB_NAME ($JOB_TYPE)"
    gantry run \
        --remote https://github.com/ai2cm/ace \
        --ref "$ACE_GIT_REF" \
        --name "$JOB_NAME" \
        --task-name "$JOB_NAME" \
        --description "ACE2.2-ERA5 with diagnostic near-surface fields ($JOB_NAME)" \
        --beaker-image "$IMAGE" \
        --workspace ai2/ace \
        --priority high \
        --min-runtime 8h \
        "${CLUSTER_ARGS[@]}" \
        --env WANDB_USERNAME=$WANDB_IDENTITY \
        --env WANDB_NAME="$JOB_NAME" \
        --env WANDB_JOB_TYPE="$JOB_TYPE" \
        --env WANDB_RUN_GROUP="$ARM" \
        --env EXPERIMENT_CONFIG_COMMIT="$CONFIG_COMMIT" \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-${BEAKER_USERNAME} \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        ${MOUNT:+--dataset "$MOUNT"} \
        --gpus $GPUS \
        --shared-memory $SHM \
        --weka climate-default:/climate-default \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- bash -c "echo '${CONFIG_B64}' | base64 -d > /tmp/config.yaml && echo '${CLIP_OFF_B64}' | base64 -d > /tmp/clip_off.py && $COMMAND"
}

# Launches JOB_NAME once DEPENDENCY (empty for none) has a successful job.
stage () {
    local JOB_NAME=$1 CONFIG_PATH=$2 DEPENDENCY=$3 MOUNT_PATH=$4 ARM=$5 JOB_TYPE=$6 COMMAND=$7
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
    launch "$JOB_NAME" "$CONFIG_PATH" "$MOUNT" "$ARM" "$JOB_TYPE" "$COMMAND"
}

# The clip-off copy is made inside the job (/ckpt.tar is read-only).
CLIP_OFF="python /tmp/clip_off.py /ckpt.tar /tmp/ckpt.tar"
EVAL="torchrun --nproc_per_node 1 -m fme.ace.evaluator /tmp/config.yaml"
for ARM in $ARMS; do
    for SEED in $SEEDS; do
        STAGE1="$ARM-stage1-rs$SEED"
        STAGE2="$ARM-stage2-rs$SEED"
        TRAIN="torchrun --nproc_per_node $TRAIN_GPUS -m fme.ace.train /tmp/config.yaml --override seed=$SEED"
        stage "$STAGE1" "$SCRIPT_DIR/${STAGE1_CONFIG[$ARM]}" "" "" "$ARM" training "$TRAIN"
        stage "$STAGE2" "$SCRIPT_DIR/multi-step-ft.yaml" "$STAGE1" /weights "$ARM" training "$TRAIN"
        stage "$ARM-long36-rs$SEED" "$SCRIPT_DIR/long36-evaluator-config.yaml" \
            "$STAGE2" "$CKPT:/ckpt.tar" "$ARM" inference "$EVAL"
        stage "$ARM-long36-clipoff-rs$SEED" "$SCRIPT_DIR/long36-evaluator-config.yaml" \
            "$STAGE2" "$CKPT:/ckpt.tar" "$ARM" inference \
            "$CLIP_OFF && $EVAL --override checkpoint_path=/tmp/ckpt.tar"
        for CLIP in on off; do
            for EXP in control p4k; do
                if [[ $CLIP == on ]]; then
                    JOB_NAME="$ARM-perturbation-$EXP-rs$SEED" PREP="true" CKPT_PATH=/ckpt.tar
                else
                    JOB_NAME="$ARM-perturbation-$EXP-clipoff-rs$SEED" PREP=$CLIP_OFF CKPT_PATH=/tmp/ckpt.tar
                fi
                # seed=1 is the inference noise draw ACE2.2-ERA5's seed runs used.
                OVERRIDE="checkpoint_path=$CKPT_PATH initial_condition.path=$IC_PATH"
                OVERRIDE="$OVERRIDE experiment_dir=$OUTPUT_ROOT/$JOB_NAME seed=1"
                stage "$JOB_NAME" "$SCRIPT_DIR/inference-$EXP.yaml" "$STAGE2" "$CKPT:/ckpt.tar" \
                    "$ARM" inference \
                    "$PREP && python -I -m fme.ace.inference /tmp/config.yaml --override $OVERRIDE"
            done
        done
    done
done
