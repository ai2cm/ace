#!/bin/bash
#
# Matched-unit moisture evaluation of the two 40-epoch arms. One
# non-preemptible job per arm runs three evaluators back to back (day-5 skill
# over 1996-1997, the 1996-2000 window, the 81-year rollout), each saving
# daily PRESsfc, air_temperature_3/7 and the level-3 moisture variable so
# specific total water and relative humidity can be compared in matching units
# offline. Uses the EMA weights of the epoch-40 checkpoint.
#
# Usage (run FROM this directory):  ./run-evaluate.sh [<substring> ...]

set -euo pipefail

SCRIPT_PATH=$(git rev-parse --show-prefix)
REPO_ROOT=$(git rev-parse --show-toplevel)
if [[ -z "$SCRIPT_PATH" ]]; then
  echo "ERROR: run from the configs directory, not the repo root." >&2
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

cd "$REPO_ROOT"

run_evaluate() {
  local arm="$1"
  local results_dataset="$2"
  local job_name="evaluate-${arm}-4deg-rs1-matched-moisture"
  should_run "$job_name" || { echo "skip (filter): $job_name"; return 0; }

  local cmd=""
  for w in skill 1996 81year; do
    local cfg="$SCRIPT_PATH/evaluate-${arm}-4deg-rs1-${w}.yaml"
    python -m fme.ace.validate_config --config_type evaluator "$cfg"
    cmd+="python -I -m fme.ace.evaluator $cfg; "
  done
  cmd+="true"

  echo "launching: $job_name"
  gantry run \
    --name "$job_name" \
    --description 'Matched-unit moisture evaluation, prognostic RH vs specific total water (4deg daily)' \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace \
    --priority "${PRIORITY:-high}" \
    --not-preemptible \
    --cluster ai2/jupiter \
    --cluster ai2/titan \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "$results_dataset:training_checkpoints/ema_ckpt_0040.tar:/ckpt.tar" \
    --gpus 1 \
    --shared-memory 50GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- bash -c "$cmd"
}

# Result datasets of the 40-epoch training jobs.
run_evaluate rh 01M74EA6VSGJGK5E25RH2H6RWN
run_evaluate corroff 01M74EA6ZA7W9F1F4VVXEBV5JK
