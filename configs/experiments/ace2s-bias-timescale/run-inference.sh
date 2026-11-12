#!/bin/bash
#
# ACE2S bias-development timescale: one evaluator job per checkpoint of the
# pvt11nqc detached fine-tune, each running evaluator-10yr-warm-years.yaml
# (8 start dates x 4 members). Run from this directory:
#   ./run-inference.sh            # both checkpoints
#   ./run-inference.sh ep40       # only jobs whose name contains "ep40"
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
  echo "ERROR: SCRIPT_PATH is empty. Run from configs/experiments/ace2s-bias-timescale/." >&2
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

CONFIG=evaluator-10yr-warm-years.yaml
[[ -f "$CONFIG" ]] || { echo "ERROR: $CONFIG not found in $(pwd)" >&2; exit 1; }

run_inference() {
  local job_name="$1" ckpt_dataset="$2" ckpt_file="$3"
  should_run "$job_name" || return 0
  echo "launching: $job_name ($ckpt_dataset:$ckpt_file)"
  (cd "$REPO_ROOT" && gantry run \
    --name "$job_name" \
    --description "ACE2S bias-development timescale: 10-year warm-year rollouts, $job_name" \
    --beaker-image "$(cat latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority high \
    --not-preemptible \
    --cluster ai2/jupiter \
    --cluster ai2/titan \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP=ace2s-bias-timescale \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "$ckpt_dataset:$ckpt_file:/ckpt.tar" \
    --gpus 1 \
    --shared-memory 50GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- python -I -m fme.ace.evaluator "${SCRIPT_PATH}${CONFIG}" </dev/null)
}

# EMA epoch 10 (the frozen paper model): EMA weights extracted from
# pvt11nqc's ckpt_0010.tar.
run_inference ace2s-bias-timescale-ep10 01M2TR2B6J9HHMXQ1S8YFJN9BD ckpt.tar
# EMA epoch 40: written by the pvt11nqc resume (beaker 01M72ZPVFQ9PTQWMJ48FG2JJJC),
# result dataset of its second job.
run_inference ace2s-bias-timescale-ep40 01M74130N098SSZC1G9JHXP8SA training_checkpoints/ema_ckpt_0040.tar
