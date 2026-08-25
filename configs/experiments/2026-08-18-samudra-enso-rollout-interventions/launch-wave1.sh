#!/bin/bash
#
# Launch the wave-1 ENSO rollout-intervention training arms.
#
# Four coupled fine-tune arms (wint5 / wint20 / hzn12 / noohc; one change each
# vs the shared corrected-lineage base, nino readout channels dropped, no
# dedicated control) and one ocean pretrain from scratch (resid). See
# make_wave1_configs.py and the reports repo's intervention matrix.
#
# Ocean init: the corrected from-scratch Samudra pretrain (OHC + surface-flux
# correctors on during pretraining), whose in/out names match these configs
# exactly -- no checkpoint surgery needed.
#
# Cost note: the baseline FT ran 4 GPUs x ~2.5 days. Budget ~250 GPU-hours per
# FT arm; hzn12's 3x window may run longer or need batch_size reduced (it is
# marked a pilot: launch it alone first if memory is a concern).
#
# Examples:
#   ARMS="hzn12" ./launch-wave1.sh                   # single arm (memory pilot)
#   ./launch-wave1.sh                                 # all four FT arms
#   ARMS="resid" ./launch-wave1.sh                    # the pretrain
#   DRY_RUN=1 ./launch-wave1.sh                       # print only

set -euo pipefail

JOB_GROUP="${JOB_GROUP:-samudra-enso-rollout-interventions-w1}"
ARMS="${ARMS:-wint5 wint20 hzn12 noohc}"
DRY_RUN="${DRY_RUN:-0}"
N_GPUS="${N_GPUS:-4}"
# urgent matches the original FT run; at high the first attempts were
# preempted mid-training.
PRIORITY="${PRIORITY:-urgent}"

ATMOS_CKPT_DATASET="${ATMOS_CKPT_DATASET:-01KJ70WK2NH4T2T4AVAAPYFSHA}"
# troya/cm4-samudra-1pct-ocean-train-using-ufs-var-subset-ohc-hdfs-correctors
OCEAN_CKPT_DATASET="${OCEAN_CKPT_DATASET:-01KW2BQ83EGZ90WZ74CZ4TJATN}"
# Normalization/centering stats the configs reference at /atmos_stats and
# /ocean_stats -- mounted exactly as the original runs mounted them (missing
# these was the first launch's failure: /ocean_stats/ocean/centering.nc).
STATS_BUNDLE_DATASET="${STATS_BUNDLE_DATASET:-01KHGYVHSX504ZBHJC223S63F0}"
FT_OCEAN_STATS_DATASET="${FT_OCEAN_STATS_DATASET:-01KXH6AFMYSRYSV6PA230Q3JG7}"

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
                --dataset "${OCEAN_CKPT_DATASET}:training_checkpoints/best_inference_ckpt.tar:/ocean_ckpt.tar"
                --dataset "${STATS_BUNDLE_DATASET}:coupled_atmosphere:/atmos_stats"
                --dataset "${FT_OCEAN_STATS_DATASET}:/ocean_stats")

  if [[ "$arm" == "resid" ]]; then
    config="${CONFIG_DIR}/resid-pretrain.yaml"
    module="fme.ace.train"
    mounts=(--dataset "${STATS_BUNDLE_DATASET}:/ocean_stats")
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
    --priority "$PRIORITY" \
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
