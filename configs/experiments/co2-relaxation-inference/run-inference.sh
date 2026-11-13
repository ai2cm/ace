#!/bin/bash
#
# 81-year AMIP-style evaluator rollouts of exoomsxr (muon-ablation-pretrain-rs1,
# 1-degree ERA5 daily) best-validation checkpoint, with and without eval-mode
# relaxation of air_temperature_0 and specific_total_water_0 toward their
# 1990-2019 ln(CO2) fit (ace#1629). Run from this directory:
#   ./run-inference.sh            # all arms
#   ./run-inference.sh tau30d     # only jobs whose name contains "tau30d"
#
# Inference has no mid-run checkpointing, so jobs are not preemptible.

set -euo pipefail

# === GUARDRAILS =============================================================
WANDB_IDENTITY="mcgibbon"

SCRIPT_PATH=$(git rev-parse --show-prefix)
REPO_ROOT=$(git rev-parse --show-toplevel)

WANDB_USERNAME=${WANDB_USERNAME:-$WANDB_IDENTITY}
if [[ "$WANDB_USERNAME" != "$WANDB_IDENTITY" ]]; then
  echo "ERROR: WANDB_USERNAME='$WANDB_USERNAME' but runs must attribute to '$WANDB_IDENTITY'." >&2
  exit 1
fi

if [[ -z "$SCRIPT_PATH" ]]; then
  echo "ERROR: SCRIPT_PATH is empty. Run from configs/experiments/co2-relaxation-inference/." >&2
  exit 1
fi

LAUNCH_FILTERS=("$@")
should_run() {
  [[ ${#LAUNCH_FILTERS[@]} -eq 0 ]] && return 0
  local f
  for f in "${LAUNCH_FILTERS[@]}"; do
    [[ "$1" == *"$f"* ]] && return 0
  done
  return 1
}
# === END GUARDRAILS =========================================================

# exoomsxr (beaker 01M5ZMXYAVE6GGE44X5F29J7XP) result dataset; best_ckpt.tar is
# the best validation loss checkpoint (EMA weights, validate_using_ema: true).
CKPT_DATASET=01M60N0WEE02V8AKF9438YF3GT
CKPT_FILE=training_checkpoints/best_ckpt.tar
# compute_co2_regression.py output for air_temperature_0 and
# specific_total_water_0 vs global_mean_co2, 1990-2019 annual means of the
# 2026-08-13 1-degree daily store.
CO2_STATS_DATASET=01M7B9E6R8JH1AM01BCSGZ80N8  # jeremym/2026-11-13-era5-1deg-co2-regression-1990-2019
if [[ "$CO2_STATS_DATASET" == __* ]]; then
  echo "ERROR: CO2_STATS_DATASET is unset." >&2
  exit 1
fi

run_inference() {
  local job_name="$1" config="$2"
  should_run "$job_name" || return 0
  [[ -f "$config" ]] || { echo "ERROR: $config not found in $(pwd)" >&2; exit 1; }
  echo "launching: $job_name ($config)"
  (cd "$REPO_ROOT" && gantry run \
    --name "$job_name" \
    --description "exoomsxr 81-year AMIP inference, CO2-regression relaxation study: $job_name" \
    --beaker-image "$(cat latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority high \
    --not-preemptible \
    --cluster ai2/titan \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP=co2-relaxation-inference \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "$CKPT_DATASET:$CKPT_FILE:/ckpt.tar" \
    --dataset "$CO2_STATS_DATASET:/co2stats" \
    --gpus 1 \
    --shared-memory 50GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- python -I -m fme.ace.evaluator "${SCRIPT_PATH}${config}" </dev/null)
}

run_inference exoomsxr-81yr-baseline evaluator-81yr.yaml
run_inference exoomsxr-81yr-relax-tau1d evaluator-81yr-relax-tau1d.yaml
run_inference exoomsxr-81yr-relax-tau5d evaluator-81yr-relax-tau5d.yaml
run_inference exoomsxr-81yr-relax-tau30d evaluator-81yr-relax-tau30d.yaml
run_inference exoomsxr-81yr-relax-tau365d evaluator-81yr-relax-tau365d.yaml
run_inference exoomsxr-81yr-relax-tau730d evaluator-81yr-relax-tau730d.yaml
