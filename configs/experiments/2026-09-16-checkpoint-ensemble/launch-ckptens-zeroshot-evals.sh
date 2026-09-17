#!/bin/bash
# Zero-shot coupled scouting evals (36-IC protocol) of the checkpoint-ensemble
# blends. The coupled evaluator takes a single ocean checkpoint path, so each
# arm mounts a pre-combined ensemble stepper checkpoint (built with
# load_stepper_ensemble + get_state; see the experiment README). Reuses the
# residfix0 scouting configs verbatim: same ICs, steps, writer, and protocol.
set -euo pipefail

ARMS="${ARMS:-ff10 ff20 ff50}"
YEARS="${YEARS:-233 246 250}"
PRIORITY="${PRIORITY:-urgent}"
# Combined ensemble checkpoints (residfixbest x pretrain0), file at root.
declare -A ARM_DATASETS
ARM_DATASETS[ff10]="01M2PBJV6K5BWE9DN6HJWT6JMV"
ARM_DATASETS[ff20]="01M2PBNF2NMRTHN471WTKWB3C7"
ARM_DATASETS[ff50]="01M2PBWR0WFP90214RMY6CK9F4"
# Standard coupled-eval atmosphere.
ATMOS_DS="${ATMOS_DS:-01KJ70WK2NH4T2T4AVAAPYFSHA}"

REPO_ROOT=$(git rev-parse --show-toplevel)
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
EVAL_CONFIG_DIR="configs/experiments/2026-08-18-samudra-enso-rollout-interventions/wave1_eval_configs/residfix0"
cd "$REPO_ROOT"

for arm in $ARMS; do
  ds="${ARM_DATASETS[$arm]:-}"
  [ -z "$ds" ] && { echo "no combined ckpt dataset for $arm" >&2; exit 1; }
  for year in $YEARS; do
    ys=$(printf "%04d" "$year")
    config="${EVAL_CONFIG_DIR}/yr${ys}.yaml"
    job="samudra-enso-ckptens-zs-${arm}-yr${ys}"
    gantry run \
      --name "$job" --task-name "$job" \
      --description "Checkpoint-ensemble ${arm}: zero-shot coupled scouting eval, year ${ys}" \
      --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
      --workspace ai2/ace --priority "$PRIORITY" --preemptible \
      --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
      --weka climate-default:/climate-default \
      --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$job" \
      --env WANDB_JOB_TYPE=evaluation --env WANDB_RUN_GROUP=samudra-enso-ckptens-zs \
      --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
      --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
      --dataset-secret google-credentials:/tmp/google_application_credentials.json \
      --dataset "${ds}:combined_${arm}.tar:/ocean_ckpt.tar" \
      --dataset "${ATMOS_DS}:training_checkpoints/best_inference_ckpt.tar:/atmos_ckpt.tar" \
      --gpus 1 --shared-memory 400GiB --budget ai2/atec-climate \
      --allow-dirty --system-python --install "pip install --no-deps ." \
      -- python -I -m fme.coupled.evaluator "$config"
  done
done
