#!/bin/bash
# Usage: ./run-train.sh <base|pr>
# Launches the PR #1519 DISCO timing A/B job from the current HEAD commit.

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
 # since we use a service account API key for wandb, we use the beaker username to set the wandb username by default
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=2

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

VARIANT="$1"
if [[ "$VARIANT" != "base" && "$VARIANT" != "pr" ]]; then
  echo "Usage: $0 <base|pr>"
  exit 1
fi

CONFIG_PATH="$SCRIPT_PATH/train-nc-sfno-all-disco.yaml"
JOB_NAME="pr1519-disco-timing-$VARIANT-$(git rev-parse --short HEAD)"
JOB_GROUP="pr1519-disco-timing"

python -m fme.ace.validate_config --config_type train "$CONFIG_PATH"

gantry run \
  --name "$JOB_NAME" \
  --task-name "$JOB_NAME" \
  --description "PR #1519 DISCO timing A/B ($VARIANT)" \
  --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
  --workspace ai2/ace \
  --priority normal \
  --cluster ai2/jupiter \
  --env WANDB_USERNAME="$WANDB_USERNAME" \
  --env WANDB_NAME="$JOB_NAME" \
  --env WANDB_JOB_TYPE=training \
  --env WANDB_RUN_GROUP="$JOB_GROUP" \
  --env WANDB_PROJECT=pr-benchmarks \
  --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset-secret google-credentials:/tmp/google_application_credentials.json \
  --gpus $N_GPUS \
  --shared-memory 200GiB \
  --weka climate-default:/climate-default \
  --budget ai2/atec-climate \
  --system-python \
  --install "pip install --no-deps ." \
  -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_PATH
