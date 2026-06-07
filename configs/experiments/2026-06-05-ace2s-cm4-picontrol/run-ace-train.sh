#!/bin/bash

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-bhenn1983}
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=4
JOB_GROUP="ace2s-cm4-picontrol"

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

run_training() {
  local config_filename="$1"
  local job_name="$2"
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"
  shift 2

  local ckpt_dataset=""
  local override_args=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --ckpt) ckpt_dataset="$2"; shift 2 ;;
      *) override_args+=("$1"); shift ;;
    esac
  done

  local ckpt_arg=()
  if [[ -n "$ckpt_dataset" ]]; then
    ckpt_arg=(--dataset "$ckpt_dataset:/weights")
  fi

  python -m fme.ace.validate_config --config_type train "$CONFIG_PATH"

  gantry run \
    --name "$job_name" \
    --task-name "$job_name" \
    --description 'Run ACE2S CM4 piControl atmosphere training' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority high \
    --preemptible \
    --cluster ai2/titan \
    --weka climate-default:/climate-default \
    --dataset jamesd/2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-stats:/statsdata \
    "${ckpt_arg[@]}" \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP="$JOB_GROUP" \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --gpus $N_GPUS \
    --shared-memory 400GiB \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_PATH \
    ${override_args:+--override "${override_args[@]}"}
}

for RS in 0 1; do
  run_training "ace-train-config-1-step-pretrain.yaml" "ace2s-cm4-picontrol-1-step-pretrain-rs${RS}" "seed=${RS}"
done

# For the finetuning stage fill in PRETRAIN_DATASETS above with the beaker dataset ids
# from the pretrain jobs, then uncomment and run the loop below

# Beaker dataset IDs for pretrain checkpoints, used to initialize finetuning (one per seed)
# PRETRAIN_DATASETS=(
#     ""  # rs0
#     ""  # rs1
# )

# for RS in 0 1; do
#   run_training "ace-train-config-multi-step-finetuning.yaml" "ace2s-cm4-picontrol-multi-step-finetuning-rs${RS}" --ckpt "${PRETRAIN_DATASETS[$RS]}" "seed=${RS}"
# done
