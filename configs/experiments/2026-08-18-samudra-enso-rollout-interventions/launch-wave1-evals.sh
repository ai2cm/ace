#!/bin/bash
# Score finished wave-1 arms: free-running coupled rollout on the scouting ICs.
# One job per (arm, year). ARM_DATASETS maps arm -> its training result dataset.
set -euo pipefail

ARMS="${ARMS:?set ARMS, e.g. ARMS=\"wint5 noohc\"}"
YEARS="${YEARS:-233 246 250}"
PRIORITY="${PRIORITY:-urgent}"
# Which checkpoint to evaluate. noohc has no best_inference_ckpt.tar: its
# checkpoint selection by inference error never fired because every inference
# was NaN (the corrections-off instability), so it is evaluated at
# best_ckpt.tar (best validation loss) -- a selection asymmetry vs the other
# arms worth noting when comparing.
CKPT_FILE="${CKPT_FILE:-best_inference_ckpt.tar}"
declare -A ARM_DATASETS
ARM_DATASETS[wint5]="${WINT5_DS:-01M0YJXAQS9HMYHNTK86KQ5ADP}"
ARM_DATASETS[noohc]="${NOOHC_DS:-01M0VY2PS9Q71WCDW0F27K5ZY8}"
ARM_DATASETS[wint20]="${WINT20_DS:-}"
ARM_DATASETS[hzn12]="${HZN12_DS:-}"

REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"

for arm in $ARMS; do
  ds="${ARM_DATASETS[$arm]:-}"
  [ -z "$ds" ] && { echo "no result dataset for $arm" >&2; exit 1; }
  for year in $YEARS; do
    ys=$(printf "%04d" "$year")
    config="${SCRIPT_PATH}/wave1_eval_configs/${arm}/yr${ys}.yaml"
    job="samudra-enso-w1eval-${arm}-yr${ys}"
    gantry run \
      --name "$job" --task-name "$job" \
      --description "Wave-1 arm ${arm}: scouting-IC coupled rollout eval, year ${ys}" \
      --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
      --workspace ai2/ace --priority "$PRIORITY" --preemptible \
      --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan --weka climate-default:/climate-default \
      --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$job" \
      --env WANDB_JOB_TYPE=evaluation --env WANDB_RUN_GROUP=samudra-enso-w1-evals \
      --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
      --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
      --dataset-secret google-credentials:/tmp/google_application_credentials.json \
      --dataset "${ds}:training_checkpoints/${CKPT_FILE}:/ckpt.tar" \
      --gpus 1 --shared-memory 400GiB --budget ai2/atec-climate \
      --allow-dirty --system-python --install "pip install --no-deps ." \
      -- bash -c "python '${SCRIPT_PATH}/make_wave1_eval_configs.py' --arms ${arm} && python -I -m fme.coupled.evaluator '$config'"
  done
done
