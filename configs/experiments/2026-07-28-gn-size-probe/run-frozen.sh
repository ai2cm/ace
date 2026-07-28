#!/bin/bash
# Frozen-GroupNorm causal test -- see gn_frozen_eval.py.
#
# Four arms in one process (16 deg capture -> 4 deg live -> 4 deg frozen ->
# 32 deg single patch), scored against fine truth on a common footprint.

set -e

JOB_NAME="gn-frozen-eval-tc-20230425"

# Resolve paths from this script's own location rather than the caller's cwd.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
REL_DIR=${SCRIPT_DIR#"$REPO_ROOT"/}
CONFIG_PATH=$REL_DIR/gn-frozen-eval.yaml
PROBE_PATH=$REL_DIR/gn_frozen_eval.py

# since we use a service account API key for wandb, we use the beaker username
# to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

IMAGE="$(cat latest_deps_only_image.txt)"

# Bundled mixture-of-experts checkpoint. The mount point must match
# `mixture_of_experts_path` in gn-frozen-eval.yaml.
MOE_DATASET=01KTCHVDHY0SATWH9E0AW2PDS6

gantry run \
    --name $JOB_NAME \
    --description 'Frozen-GroupNorm causal test for downscaling extent bias' \
    --workspace ai2/climate-titan \
    --priority normal \
    --cluster ai2/titan \
    --beaker-image $IMAGE \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-annak \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset $MOE_DATASET:bundled_moe_multivariate.ckpt:/moe/bundled_moe_multivariate.ckpt \
    --weka climate-default:/climate-default \
    --gpus 1 \
    --shared-memory 400GiB \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- python $PROBE_PATH $CONFIG_PATH
