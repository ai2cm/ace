#!/bin/bash
# Evaluation of the four fine-tuned arms, each from best_inference_ckpt.tar of
# its stage-2 result dataset (mounted at /ckpt.tar), on 1 GPU per run:
#   1pctco2              the full 140-year ramp from 0001-01-02
#   picontrol            the 40-year tail 0311-0351, as in the 2026-09-29 round
#   randco2-4xco2        the held-out random-CO2 member ic_0003 at 4xCO2
# Configs are shared by the two source sets of an arm (cm4-control-*,
# cm4-masked-naive-*). The 1pctco2 and picontrol runs write to
# $GCS_ROOT/<arm>-<scenario>; the rest to the job's /results.
#
# Usage:
#   ./run-ace-evaluator.sh              # all 12 runs
#   ./run-ace-evaluator.sh randco2-4x   # optional substring filter on the job name
#
# Dry run (no config edit needed; the evaluator takes --override dotlists):
#   OVERRIDE="n_forward_steps=100 experiment_dir=/results" ./run-ace-evaluator.sh masked-naive-pic-1pct-1pctco2
# Job names then get a -dry suffix (set JOB_SUFFIX for another). DRY_RUN=1
# prints the gantry commands instead of submitting them.

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
WANDB_USERNAME=${WANDB_USERNAME:-bhenn1983}
REPO_ROOT=$(git rev-parse --show-toplevel)
JOB_GROUP="ace2s-snow-scenario-train"
CLUSTER="${CLUSTER:-ai2/jupiter}"
GCS_ROOT="gs://vcm-ml-intermediate/2026-10-06-ace2s-snow-scenario-train"
SELECT="${1:-}"

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

run_evaluator() {
  local arm="$1"
  local ckpt_dataset="$2"
  local scenario="$3"
  local min_runtime="$4"
  local priority="$5"
  local job_name="ace2s-snowscen-${arm}-${scenario}-eval"
  local config_path="${SCRIPT_PATH}${arm%-pic-1pct*}-${scenario}-evaluator.yaml"
  local experiment_dir="/results"
  case "$scenario" in
    1pctco2|picontrol) experiment_dir="$GCS_ROOT/${arm}-${scenario}" ;;
  esac
  local override_args=(--override "experiment_dir=$experiment_dir")
  local suffix="${JOB_SUFFIX:-}"
  if [ -n "${OVERRIDE:-}" ]; then
    suffix="${JOB_SUFFIX--dry}"
    override_args+=($OVERRIDE)
  fi
  job_name="${job_name}${suffix}"

  if [ -n "$SELECT" ] && [[ "$job_name" != *"$SELECT"* ]]; then
    return 0
  fi

  python -m fme.ace.validate_config --config_type evaluator "$config_path"

  local cmd=(gantry run
    --name "$job_name"
    --task-name "$job_name"
    --description "ACE2S snow scenario evaluation: $arm on $scenario"
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)"
    --workspace ai2/ace
    --priority "$priority"
    --min-runtime "$min_runtime"
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
    -- python -I -m fme.ace.evaluator "$config_path" "${override_args[@]}")

  if [[ -n "${DRY_RUN:-}" ]]; then
    printf '%q ' "${cmd[@]}"; echo
  else
    "${cmd[@]}"
  fi
}

run_arm() {
  local arm="$1"
  local ckpt_dataset="$2"
  # The ramp (~11 GPU-h plus writes) outlives the 8h min-runtime cap, so it runs
  # urgent to make preemption unlikely; the rest finish inside the cap.
  run_evaluator "$arm" "$ckpt_dataset" 1pctco2 8h urgent
  run_evaluator "$arm" "$ckpt_dataset" picontrol 8h normal
  run_evaluator "$arm" "$ckpt_dataset" randco2-4xco2 8h normal
}

# Stage-2 result datasets (README); best_inference_ckpt.tar of each.
run_arm cm4-control-pic-1pct              01M4RE0QHT2DE8V0QRCEWM43VA
run_arm cm4-masked-naive-pic-1pct         01M4J32EDGBPFNG7W8PSANBFQN
run_arm cm4-control-pic-1pct-randco2      01M4J32RBKK133V679PS5X2CWC
run_arm cm4-masked-naive-pic-1pct-randco2 01M4RZQ6E9Z8TZ8DF9BVG88E5K
