#!/bin/bash
#
# Launch the wave-1 ENSO rollout-intervention training arms.
#
# Five coupled fine-tune arms (ctrl / wint5 / wint20 / hzn12 / ohc; one change
# each vs the archived baseline run config, all with the nino readout channels
# dropped) and one ocean pretrain from scratch (resid). See
# make_wave1_configs.py and the reports repo's intervention matrix.
#
# Prerequisite (one-time): the trimmed ocean checkpoint. Run trim_ocean_ckpt.py
# per its docstring and set TRIMMED_OCEAN_CKPT_DATASET to the resulting Beaker
# dataset id. The FT arms refuse to launch without it.
#
# Cost note: the baseline FT ran 4 GPUs x ~2.5 days. Budget ~250 GPU-hours per
# FT arm; hzn12's 3x window may run longer or need batch_size reduced (it is
# marked a pilot: launch it alone first if memory is a concern).
#
# Examples:
#   ARMS="ctrl" ./launch-wave1.sh                    # single arm
#   ./launch-wave1.sh                                 # all five FT arms
#   ARMS="resid" ./launch-wave1.sh                    # the pretrain
#   DRY_RUN=1 ./launch-wave1.sh                       # print only

set -euo pipefail

JOB_GROUP="${JOB_GROUP:-samudra-enso-rollout-interventions-w1}"
ARMS="${ARMS:-ctrl wint5 wint20 hzn12 ohc}"
DRY_RUN="${DRY_RUN:-0}"
N_GPUS="${N_GPUS:-4}"

ATMOS_CKPT_DATASET="${ATMOS_CKPT_DATASET:-01KJ70WK2NH4T2T4AVAAPYFSHA}"
TRIMMED_OCEAN_CKPT_DATASET="${TRIMMED_OCEAN_CKPT_DATASET:-}"

REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
CONFIG_DIR="${SCRIPT_PATH}/wave1_configs"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')

cd "$REPO_ROOT"

launch() {
  local arm="$1"
  local job_name="samudra-enso-w1-${arm}"
  local config="${CONFIG_DIR}/${arm}.yaml"
  local module="fme.coupled.train"
  local mounts=(--dataset "${ATMOS_CKPT_DATASET}:training_checkpoints/best_inference_ckpt.tar:/atmos_ckpt.tar"
                --dataset "${TRIMMED_OCEAN_CKPT_DATASET}:/ocean_ckpt.tar")

  if [[ "$arm" == "resid" ]]; then
    config="${CONFIG_DIR}/resid-pretrain.yaml"
    module="fme.ace.train"
    mounts=()
  elif [[ -z "$TRIMMED_OCEAN_CKPT_DATASET" && "$DRY_RUN" != "1" ]]; then
    echo "TRIMMED_OCEAN_CKPT_DATASET is not set; run trim_ocean_ckpt.py first" >&2
    exit 1
  fi

  python "${SCRIPT_PATH}/make_wave1_configs.py" >/dev/null

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [dry-run] $job_name ($module, $config)"
    return
  fi

  gantry run \
    --name "$job_name" \
    --task-name "$job_name" \
    --description "ENSO rollout intervention wave 1, arm ${arm}" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/climate-titan \
    --priority high \
    --preemptible \
    --cluster ai2/titan \
    --weka climate-default:/climate-default \
    --env WANDB_USERNAME="$BEAKER_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP="$JOB_GROUP" \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    "${mounts[@]}" \
    --gpus "$N_GPUS" \
    --shared-memory 600GiB \
    --budget ai2/atec-climate \
    --allow-dirty \
    --system-python \
    --install "pip install --no-deps ." \
    -- bash -c \
        "python '${SCRIPT_PATH}/make_wave1_configs.py' && \
         torchrun --nproc_per_node=${N_GPUS} -m ${module} '${config}'"
}

for arm in $ARMS; do
  launch "$arm"
done
