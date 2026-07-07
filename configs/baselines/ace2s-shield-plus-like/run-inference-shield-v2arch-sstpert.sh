#!/bin/bash
# Offline SST-perturbation inference for the ACE2S-SHiELD+ 4-deg/daily "v2arch"
# checkpoint (wandb 2ijn8ynx, last-epoch EMA ckpt ema_ckpt_0240.tar).
# Two runs over the SHiELD AMIP forcing, identical except a uniform SST offset:
#   +0 K  and  +4 K   (constant SST perturbation).
# Default launch target: ai2/ace workspace, high priority, jupiter+titan.
set -euo pipefail

WANDB_IDENTITY="mcgibbon"
JOB_GROUP="ace-shield-v2arch-sstpert"

SCRIPT_PATH=$(git rev-parse --show-prefix)   # repo-root-relative dir of this script
REPO_ROOT=$(git rev-parse --show-toplevel)
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')

# WANDB attribution guard: the beaker job env does not carry WANDB_USERNAME and the
# beaker account (jeremym) makes wandb fall back to the service account, so an
# unset/wrong value silently misattributes the run. Beaker specs are immutable.
WANDB_USERNAME=${WANDB_USERNAME:-$WANDB_IDENTITY}
if [[ "$WANDB_USERNAME" != "$WANDB_IDENTITY" ]]; then
  echo "ERROR: WANDB_USERNAME='$WANDB_USERNAME' but runs must attribute to '$WANDB_IDENTITY'." >&2
  echo "       (BEAKER_USERNAME='$BEAKER_USERNAME' would misattribute to the wandb service account.)" >&2
  exit 1
fi

# cwd / path guard.
if [[ -z "$SCRIPT_PATH" ]]; then
  echo "ERROR: SCRIPT_PATH (git rev-parse --show-prefix) is empty." >&2
  echo "       Invoke this script FROM its own configs directory, not the repo root." >&2
  exit 1
fi

P0K="$SCRIPT_PATH/ace-inference-shield-v2arch-p0k.yaml"
P4K="$SCRIPT_PATH/ace-inference-shield-v2arch-p4k.yaml"

cd "$REPO_ROOT"
python -m fme.ace.validate_config --config_type inference "$P0K"
python -m fme.ace.validate_config --config_type inference "$P4K"

# last-epoch EMA checkpoint of the v2arch run (wandb 2ijn8ynx, epoch 240).
CKPT_MOUNT="01KWQSYAEMVSYGQ4AD5NH4XK5P:training_checkpoints/ema_ckpt_0240.tar:/ckpt.tar"

# launch_job <job_name> <config_path>
launch_job () {
    JOB_NAME=$1
    CONFIG_PATH=$2
    cd "$REPO_ROOT" && gantry run \
        --yes \
        --no-logs \
        --name "$JOB_NAME" \
        --task-name "$JOB_NAME" \
        --description 'ACE2S-SHiELD+ v2arch SST-pert offline inference' \
        --beaker-image "$(cat "$REPO_ROOT"/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        --cluster ai2/jupiter \
        --cluster ai2/titan \
        --env WANDB_USERNAME="$WANDB_USERNAME" \
        --env WANDB_NAME="$JOB_NAME" \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP="$JOB_GROUP" \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset "$CKPT_MOUNT" \
        --gpus 1 \
        --shared-memory 50GiB \
        --allow-dirty \
        --weka climate-default:/climate-default \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- python -I -m fme.ace.inference "$CONFIG_PATH"
}

launch_job "shield-v2arch-sstpert-p0k" "$P0K"
launch_job "shield-v2arch-sstpert-p4k" "$P4K"
