#!/bin/bash
#
# Restoring-force experiment for exoomsxr (muon-ablation-pretrain-rs1, 1-degree
# ERA5 daily): does each checkpoint, started from a state on the other
# checkpoint's (or ERA5's) climate, move back toward its own climate, and how
# fast? Checkpoints: best validation (epoch 120, drifts to a biased climate in
# long rollouts) and EMA epoch 20 (does not).
#   stage A  states-<ckpt>                   5-year rollout saving the full
#                                            prognostic state every 73 days
#   stage B  <ckpt>-from-<source>-states-cN  30-day runs from stage A states
#   stage C  <ckpt>-from-era5-cN             30-day runs from ERA5 states on
#                                            the same dates
# Run from this directory:
#   ./run-inference.sh restore-states    # stage A
#   ./run-inference.sh from-era5         # stage C
#   ./run-inference.sh -states-c         # stage B (needs IC_DATASET)
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
  echo "ERROR: SCRIPT_PATH is empty. Run from configs/experiments/restoring-force/." >&2
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

# exoomsxr (beaker 01M5ZMXYAVE6GGE44X5F29J7XP) result dataset.
CKPT_DATASET=01M60N0WEE02V8AKF9438YF3GT
BEST_CKPT=training_checkpoints/best_ckpt.tar
EMA20_CKPT=training_checkpoints/ema_ckpt_0020.tar
# Stage A states restacked to one leading sample dimension: ep120-states.nc and
# ep20-states.nc, 200 states each (IC-major, snapshot-minor).
IC_DATASET=__FILL_AFTER_STAGE_A__

# run_inference <job_name> <module> <config> <ckpt_file> [needs_ic]
run_inference() {
  local job_name="$1" module="$2" config="$3" ckpt_file="$4" needs_ic="${5:-}"
  should_run "$job_name" || return 0
  [[ -f "$config" ]] || { echo "ERROR: $config not found in $(pwd)" >&2; exit 1; }
  local extra=()
  if [[ -n "$needs_ic" ]]; then
    if [[ "$IC_DATASET" == __* ]]; then
      echo "ERROR: IC_DATASET is unset; $job_name needs the stage A states." >&2
      exit 1
    fi
    extra=(--dataset "$IC_DATASET:/ic")
  fi
  echo "launching: $job_name ($module, $config, $ckpt_file)"
  (cd "$REPO_ROOT" && gantry run \
    --name "$job_name" \
    --description "exoomsxr restoring-force experiment: $job_name" \
    --beaker-image "$(cat latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority high \
    --not-preemptible \
    --cluster ai2/titan \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP=restoring-force \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "$CKPT_DATASET:$ckpt_file:/ckpt.tar" \
    "${extra[@]}" \
    --gpus 1 \
    --shared-memory 50GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- python -I -m "$module" "${SCRIPT_PATH}${config}" </dev/null)
}

# stage A
run_inference restore-states-ep120 fme.ace.evaluator evaluator-states.yaml "$BEST_CKPT"
run_inference restore-states-ep20 fme.ace.evaluator evaluator-states.yaml "$EMA20_CKPT"

for i in 0 1 2 3; do
  # stage C
  run_inference "restore-ep120-from-era5-c$i" fme.ace.evaluator "evaluator-era5-states-c$i.yaml" "$BEST_CKPT"
  run_inference "restore-ep20-from-era5-c$i" fme.ace.evaluator "evaluator-era5-states-c$i.yaml" "$EMA20_CKPT"
  # stage B
  for source in ep120 ep20; do
    run_inference "restore-ep120-from-$source-states-c$i" fme.ace.inference "inference-$source-states-c$i.yaml" "$BEST_CKPT" ic
    run_inference "restore-ep20-from-$source-states-c$i" fme.ace.inference "inference-$source-states-c$i.yaml" "$EMA20_CKPT" ic
  done
done
