#!/bin/bash

set -e
export GRPC_VERBOSITY=ERROR

# Seed replicates of im4ecamc (wandb), Jeremy's 4-degree no-CO2 model with ACE2.2's recipe, to
# measure the seed spread of its in-sample trend against the ACE2S recipe's seeds in
# ../2026-10-23-ace25-aimip-4deg/. Single stage; seed 0 is im4ecamc itself.
#
# Safe to rerun at any point: each seed is launched only if its beaker experiment does not exist.
#
#   bash run.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NAME_PREFIX="im4ecamc-recipe-4deg"
SEEDS="1 2 3"

# The code and image im4ecamc was trained with (its config is train.yaml, copied verbatim from
# configs/baselines/aimip-like/ at this commit).
ACE_GIT_REF="d9faa918547b4646497d1107845d6b67cdde5dd4"
IMAGE="elynn/fme-deps-only-769435616"

CONFIG_COMMIT=$(git -C "$SCRIPT_DIR" rev-parse HEAD)
if [[ -n "$(git -C "$SCRIPT_DIR" -c core.fileMode=false status --porcelain -- .)" ]]; then
    CONFIG_COMMIT="$CONFIG_COMMIT-dirty"
fi
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_IDENTITY="bhenn1983"  # differs from BEAKER_USERNAME; do not derive one from the other

experiment_exists () {
    beaker experiment get "$BEAKER_USERNAME/$1" >/dev/null 2>&1
}

launch () {
    local JOB_NAME=$1 SEED=$2
    local CONFIG_B64
    CONFIG_B64=$(base64 < "$SCRIPT_DIR/train.yaml" | tr -d '\n')
    echo "launch $JOB_NAME"
    gantry run \
        --remote https://github.com/ai2cm/ace \
        --ref "$ACE_GIT_REF" \
        --name "$JOB_NAME" \
        --task-name "$JOB_NAME" \
        --description "im4ecamc recipe seed replicate ($JOB_NAME)" \
        --beaker-image "$IMAGE" \
        --workspace ai2/ace \
        --priority high \
        --min-runtime 8h \
        --cluster ai2/jupiter-cirrascale-2 \
        --cluster ai2/titan-cirrascale \
        --env WANDB_USERNAME=$WANDB_IDENTITY \
        --env WANDB_NAME="$JOB_NAME" \
        --env WANDB_JOB_TYPE=training \
        --env WANDB_RUN_GROUP="$NAME_PREFIX" \
        --env EXPERIMENT_CONFIG_COMMIT="$CONFIG_COMMIT" \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-${BEAKER_USERNAME} \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --gpus 1 \
        --shared-memory 400GiB \
        --weka climate-default:/climate-default \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- bash -c "echo '${CONFIG_B64}' | base64 -d > /tmp/config.yaml && torchrun --nproc_per_node 1 -m fme.ace.train /tmp/config.yaml --override seed=${SEED}"
}

for SEED in $SEEDS; do
    JOB_NAME="$NAME_PREFIX-rs$SEED"
    if experiment_exists "$JOB_NAME"; then
        echo "skip $JOB_NAME: experiment exists"
        continue
    fi
    launch "$JOB_NAME" "$SEED"
done
