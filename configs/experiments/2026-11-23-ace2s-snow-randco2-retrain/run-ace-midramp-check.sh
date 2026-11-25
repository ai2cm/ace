#!/bin/bash
# Mid-ramp check of the epoch-1 fine-tune checkpoints: 30 years of the 1pctCO2
# ramp from the CM4 state of 0041-01-01 (cm4-*-midramp-evaluator.yaml), on 1 GPU
# per run, writing to $GCS_ROOT/<run>-ep1-midramp. Each checkpoint dataset holds
# training_checkpoints/best_inference_ckpt.tar after the first fine-tune epoch.
# Their full-ramp evaluations jumped (or did not) in ramp years 50-68; this asks
# whether the same jumps appear without the first 40 years of free running.
#
#   ./run-ace-midramp-check.sh           # all six runs
#   ./run-ace-midramp-check.sh randco2   # optional substring filter on the job name
#   DRY_RUN=1 ./run-ace-midramp-check.sh # print the gantry commands only

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)
WANDB_USERNAME=${WANDB_USERNAME:-bhenn1983}
REPO_ROOT=$(git rev-parse --show-toplevel)
JOB_GROUP="ace2s-snow-randco2-retrain"
CLUSTER="${CLUSTER:-ai2/jupiter}"
GCS_ROOT="gs://vcm-ml-intermediate/2026-11-23-ace2s-snow-randco2-retrain"
SELECT="${1:-}"

cd $REPO_ROOT

run_midramp() {
  local run="$1-ep1"
  local ckpt_dataset="$2"
  local job_name="ace2s-snowfix-${run}-midramp-eval"
  local config_path="${SCRIPT_PATH}${run%-pic-1pct*}-midramp-evaluator.yaml"

  if [ -n "$SELECT" ] && [[ "$job_name" != *"$SELECT"* ]]; then
    return 0
  fi

  python -m fme.ace.validate_config --config_type evaluator "$config_path"

  local cmd=(gantry run
    --name "$job_name"
    --task-name "$job_name"
    --description "ACE2S snow random-CO2 retrain mid-ramp check: $run"
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)"
    --workspace ai2/ace
    --priority normal
    --min-runtime 2h
    --cluster "$CLUSTER"
    --weka climate-default:/climate-default
    --env WANDB_USERNAME="$WANDB_USERNAME"
    --env WANDB_NAME="$job_name"
    --env WANDB_JOB_TYPE=inference
    --env WANDB_RUN_GROUP="$JOB_GROUP"
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa
    --dataset-secret google-credentials:/tmp/google_application_credentials.json
    --dataset "$ckpt_dataset:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar"
    --gpus 1
    --shared-memory 50GiB
    --budget ai2/atec-climate
    --system-python
    --install "pip install --no-deps ."
    -- python -I -m fme.ace.evaluator "$config_path" --override "experiment_dir=$GCS_ROOT/${run}-midramp")

  if [[ -n "${DRY_RUN:-}" ]]; then
    printf '%q ' "${cmd[@]}"; echo
  else
    "${cmd[@]}"
  fi
}

# Epoch-1 checkpoint datasets (README).
run_midramp cm4-masked-naive-pic-1pct-rs1          01M88NNKYM6B020BRCVN31G0F2
run_midramp cm4-control-pic-1pct-rs1               01M88NNKWKQJXEYMY3YJDY5DB7
run_midramp cm4-masked-naive-pic-1pct-randco2-rs0  01M88YKAEXTD23HBTS1NQACKH9
run_midramp cm4-control-pic-1pct-randco2-rs0       01M88TCFV7GJW4QR5TMC9J73Z4
run_midramp cm4-masked-naive-pic-1pct-randco2-rs1  01M88YK5JMF8T8THKYXS3VK8QV
run_midramp cm4-control-pic-1pct-randco2-rs1       01M89AZR1T1PFW8VEJZRCQVJQY
