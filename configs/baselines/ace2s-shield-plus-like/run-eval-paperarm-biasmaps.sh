#!/bin/bash
#
# Launch the 3 evaluator jobs that produce time_mean_diagnostics.nc (gen_map /
# target_map / bias_map) for the 4deg/daily ACE2S-SHiELD+ paper-arm checkpoint
# (wandb f3fhd6hn), for the report's bias-map side-by-side vs the paper's
# 1-degree maps:
#   amip-test   AMIP historical, 2012-2020 test window (Fig 2c-f analog)
#   som-1xco2   SOM 1xCO2 equilibrium (Fig 3 / S3 analog)
#   som-4xco2   SOM 4xCO2 equilibrium
#
# Usage (run FROM this configs directory, not the repo root):
#   ./run-eval-paperarm-biasmaps.sh                # launch all 3
#   ./run-eval-paperarm-biasmaps.sh som            # only jobs matching "som"
set -euo pipefail

# === GUARDRAILS (from run-inference-4k-response.sh) ===
WANDB_IDENTITY="mcgibbon"
SCRIPT_PATH=$(git rev-parse --show-prefix)
REPO_ROOT=$(git rev-parse --show-toplevel)
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')

WANDB_USERNAME=${WANDB_USERNAME:-$WANDB_IDENTITY}
if [[ "$WANDB_USERNAME" != "$WANDB_IDENTITY" ]]; then
  echo "ERROR: WANDB_USERNAME='$WANDB_USERNAME' but runs must attribute to '$WANDB_IDENTITY'." >&2
  echo "       (BEAKER_USERNAME='$BEAKER_USERNAME' would misattribute to the wandb service account.)" >&2
  echo "       Run:  export WANDB_USERNAME=$WANDB_IDENTITY   before launching." >&2
  exit 1
fi
if [[ -z "$SCRIPT_PATH" ]]; then
  echo "ERROR: SCRIPT_PATH (git rev-parse --show-prefix) is empty." >&2
  echo "       Invoke this script FROM its own configs directory, not the repo root." >&2
  exit 1
fi

LAUNCH_FILTERS=("$@")
should_run() {
  [[ ${#LAUNCH_FILTERS[@]} -eq 0 ]] && return 0
  local f
  for f in "${LAUNCH_FILTERS[@]}"; do
    [[ "$1" == *"$f"* || "$2" == *"$f"* ]] && return 0
  done
  return 1
}
# === END GUARDRAILS =========================================================

JOB_GROUP="ace2s-shield-plus-4deg-daily-paperarm-biasmaps"

# best_inference_ckpt of the DONE paper-arm training run (wandb f3fhd6hn),
# final result dataset of beaker 01KWFVPCPJDVEAYDQ64431DRJR.
CKPT="01KWWQGQGRCFGQMVD6E9WVWGNJ"

cd "$REPO_ROOT"

for cfg in eval-paperarm-amip-test.yaml eval-paperarm-som-1xco2.yaml eval-paperarm-som-4xco2.yaml; do
  python -m fme.ace.validate_config --config_type evaluator "$SCRIPT_PATH/$cfg"
done

launch_job () {
  local job_name="$1"
  local config_filename="$2"
  local config_path="$SCRIPT_PATH/$config_filename"
  should_run "$config_filename" "$job_name" || { echo "skip (filter): $job_name"; return 0; }
  if [[ ! -f "$config_path" ]]; then
    echo "ERROR: config not found: $REPO_ROOT/$config_path" >&2
    exit 1
  fi
  echo "launching: $job_name  ($config_path, ckpt $CKPT)"
  cd "$REPO_ROOT" && gantry run \
    --name "$job_name" \
    --task-name "$job_name" \
    --description '4deg/daily ACE2S-SHiELD+ paper-arm evaluator: time-mean bias maps' \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace \
    --priority high \
    --not-preemptible \
    --cluster ai2/jupiter \
    --cluster ai2/titan \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=inference \
    --env WANDB_RUN_GROUP="$JOB_GROUP" \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "${CKPT}:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar" \
    --gpus 1 \
    --shared-memory 50GiB \
    --allow-dirty \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- python -I -m fme.ace.evaluator "$config_path"
}

launch_job "${JOB_GROUP}-amip-test" "eval-paperarm-amip-test.yaml"
launch_job "${JOB_GROUP}-som-1xco2" "eval-paperarm-som-1xco2.yaml"
launch_job "${JOB_GROUP}-som-4xco2" "eval-paperarm-som-4xco2.yaml"
