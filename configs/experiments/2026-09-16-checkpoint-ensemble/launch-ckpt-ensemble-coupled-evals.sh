#!/bin/bash
# Zero-shot coupled ENSO skill evals of the checkpoint-ensemble blends.
# The coupled evaluator does not take a list-valued checkpoint_path, so each
# blend is a pre-combined single stepper checkpoint (ensemble step inside),
# built with load_stepper_ensemble + get_state and uploaded to BLEND_DS.
# Protocol = the standard zero-shot coupled eval (zeroshot240 configs,
# 12 ICs x 146 coupled steps per year); scouting years by default.
set -euo pipefail

ARMS="${ARMS:-ff10 ff20 ff50}"
YEARS="${YEARS:-233 246 250}"
PRIORITY="${PRIORITY:-urgent}"
BLEND_DS="${BLEND_DS:-01M2R5JMPMBG93DGP7X879P93Y}"
ATMOS_DS="${ATMOS_DS:-01KJ70WK2NH4T2T4AVAAPYFSHA}"

REPO_ROOT=$(git rev-parse --show-toplevel)
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
EVAL_CONFIG_DIR="configs/experiments/2026-08-18-samudra-enso-rollout-interventions/wave1_eval_configs/zeroshot240"
cd "$REPO_ROOT"

for arm in $ARMS; do
  for year in $YEARS; do
    ys=$(printf "%04d" "$year")
    config="${EVAL_CONFIG_DIR}/yr${ys}.yaml"
    job="samudra-enso-ckptens-cpl-${arm}-yr${ys}"
    gantry run \
      --name "$job" --task-name "$job" \
      --description "Checkpoint-ensemble blend ${arm}: zero-shot coupled scouting eval, year ${ys}" \
      --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
      --workspace ai2/ace --priority "$PRIORITY" --preemptible \
      --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
      --weka climate-default:/climate-default \
      --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$job" \
      --env WANDB_JOB_TYPE=evaluation --env WANDB_RUN_GROUP=samudra-enso-ckptens-cpl \
      --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
      --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
      --dataset-secret google-credentials:/tmp/google_application_credentials.json \
      --dataset "${BLEND_DS}:blend_${arm}_ckpt.tar:/ocean_ckpt.tar" \
      --dataset "${ATMOS_DS}:training_checkpoints/best_inference_ckpt.tar:/atmos_ckpt.tar" \
      --gpus 1 --shared-memory 400GiB --budget ai2/atec-climate \
      --allow-dirty --system-python --install "pip install --no-deps ." \
      -- python -I -m fme.coupled.evaluator "$config"
  done
done
