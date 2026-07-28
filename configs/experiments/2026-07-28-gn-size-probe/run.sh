#!/bin/bash
# GroupNorm extent-sensitivity probe -- see gn_probe.py.
#
# Diagnostic only: no training, no wandb logging, single GPU. Reads the bundled
# MoE checkpoint and writes per-extent GroupNorm statistics to /results.

set -e

JOB_NAME="gn-size-probe-tc-20230425"

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')
CONFIG_PATH=$SCRIPT_PATH/gn-probe.yaml
PROBE_PATH=$SCRIPT_PATH/gn_probe.py

# since we use a service account API key for wandb, we use the beaker username
# to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

IMAGE="$(cat latest_deps_only_image.txt)"

# Bundled mixture-of-experts checkpoint. The mount point here must match
# `mixture_of_experts_path` in gn-probe.yaml.
MOE_DATASET=01KTCHVDHY0SATWH9E0AW2PDS6

gantry run \
    --name $JOB_NAME \
    --description 'Probe GroupNorm statistics vs input spatial extent' \
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
