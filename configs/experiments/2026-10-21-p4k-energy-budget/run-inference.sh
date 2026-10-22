#!/bin/bash

set -e
export GRPC_VERBOSITY=ERROR

# Control and +4 K AIMIP rollouts of three ACE models, writing the surface energy and
# moisture budget terms. Each model runs at the ace ref and beaker image it was last run
# under, because ACE2.1's checkpoint does not load on current code. Jobs whose beaker
# experiment already exists are skipped, so the script is safe to rerun.
#
# Usage (from this directory or anywhere):
#   bash run-inference.sh                       # all 6 jobs
#   MODELS="ace22" EXPERIMENTS="control" bash run-inference.sh
#   SMOKE=1 MODELS="ace21 ace22 ace21ec" EXPERIMENTS="control" bash run-inference.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODELS="${MODELS:-ace21 ace22 ace21ec}"
EXPERIMENTS="${EXPERIMENTS:-control p4k}"
SMOKE="${SMOKE:-}"

declare -A CHECKPOINT=(
    [ace21]="01KAKXY0EK24K7BZK2N8SPJ5SJ"    # ACE2.1-ERA5 AIMIP submission, pressure-level FT
    [ace22]="01M1WBBJDFYA9VS46V6HH3K1MW"    # ACE2.2-ERA5 AIMIP submission, seed 3, stage 3
    [ace21ec]="01KSYDZ80YW63MQ0QN0DR5F4VD"  # ACE2.1 recipe + energy corrector, seed 0, pressure-level FT
)
declare -A ACE_REF=(
    # Tree-identical to ai2cm/full-model c8295c4, which ACE2.1's AIMIP inference ran at.
    [ace21]="ea505f1b75aa1236968ee63ecf6c518ef406281a"
    [ace22]="394e41b5e07d1a3606d51faea1bd09a6fbddb492"
    [ace21ec]="59a8a3e46b6b71b9e59afa18790c3ba99c0f1899"
)
declare -A IMAGE=(
    [ace21]="01K97XSZ8NJKEZ4YM8BN13XDCZ"
    [ace22]="01KZ76KEV0T2DN5G712NRGG1SQ"
    [ace21ec]="01KS0HKT272A104Y831YXRD949"
)
# Matches each model's AIMIP production member r1. ACE2.2 is stochastic; the others ignore it.
declare -A MODEL_OVERRIDE=(
    [ace21]=""
    [ace22]="seed=1"
    [ace21ec]=""
)

NAME_PREFIX="p4k-energy-budget"
OUTPUT_ROOT="/climate-default/2026-10-21-p4k-energy-budget"
MIN_RUNTIME="8h"
if [[ -n "$SMOKE" ]]; then
    NAME_PREFIX="$NAME_PREFIX-smoke"
    OUTPUT_ROOT="$OUTPUT_ROOT-smoke"
    MIN_RUNTIME="1h"
fi

# The commit holding these configs, recorded on each job; jobs run the model's own ref,
# so beaker's git ref does not point here. "-dirty" marks uncommitted changes in this
# directory at launch time.
CONFIG_COMMIT=$(git -C "$SCRIPT_DIR" rev-parse HEAD)
if [[ -n "$(git -C "$SCRIPT_DIR" status --porcelain -- .)" ]]; then
    CONFIG_COMMIT="$CONFIG_COMMIT-dirty"
fi

BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_IDENTITY="bhenn1983"  # differs from BEAKER_USERNAME; do not derive one from the other

for MODEL in $MODELS; do
    for EXPERIMENT in $EXPERIMENTS; do
        CONFIG_PATH="$SCRIPT_DIR/$MODEL-$EXPERIMENT.yaml"
        JOB_NAME="$NAME_PREFIX-$MODEL-$EXPERIMENT"
        if beaker experiment get "$BEAKER_USERNAME/$JOB_NAME" >/dev/null 2>&1; then
            echo "skip $JOB_NAME: experiment exists"
            continue
        fi
        OVERRIDE="experiment_dir=$OUTPUT_ROOT/$JOB_NAME ${MODEL_OVERRIDE[$MODEL]}"
        [[ -n "$SMOKE" ]] && OVERRIDE="$OVERRIDE n_forward_steps=400"
        CONFIG_B64=$(base64 < "$CONFIG_PATH" | tr -d '\n')
        echo "launch $JOB_NAME at ${ACE_REF[$MODEL]:0:9}: $OVERRIDE"

        gantry run \
            --remote https://github.com/ai2cm/ace \
            --ref "${ACE_REF[$MODEL]}" \
            --name "$JOB_NAME" \
            --task-name "$JOB_NAME" \
            --description "Control/+4 K AIMIP rollout with surface energy budget outputs ($MODEL)" \
            --beaker-image "${IMAGE[$MODEL]}" \
            --workspace ai2/ace \
            --priority high \
            --min-runtime "$MIN_RUNTIME" \
            --cluster ai2/titan-cirrascale \
            --cluster ai2/jupiter-cirrascale-2 \
            --env WANDB_USERNAME=$WANDB_IDENTITY \
            --env WANDB_NAME="$JOB_NAME" \
            --env WANDB_JOB_TYPE=inference \
            --env WANDB_RUN_GROUP=p4k-energy-budget \
            --env EXPERIMENT_CONFIG_COMMIT="$CONFIG_COMMIT" \
            --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
            --env-secret WANDB_API_KEY=wandb-api-key-${BEAKER_USERNAME} \
            --dataset-secret google-credentials:/tmp/google_application_credentials.json \
            --dataset "${CHECKPOINT[$MODEL]}:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar" \
            --gpus 1 \
            --shared-memory 50GiB \
            --weka climate-default:/climate-default \
            --budget ai2/atec-climate \
            --system-python \
            --install "pip install --no-deps ." \
            -- bash -c "echo '${CONFIG_B64}' | base64 -d > /tmp/config.yaml && python -I -m fme.ace.inference /tmp/config.yaml --override ${OVERRIDE}"
    done
done
