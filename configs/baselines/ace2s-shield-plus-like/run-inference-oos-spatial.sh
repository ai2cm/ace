#!/bin/bash
#
# Launch the 2 prescribed-SST OUT-OF-SAMPLE inference jobs for the SHiELD-AMIP
# spatial near-surface-T tracking evaluation:
#   2 checkpoints (with-CO2 / no-CO2)  x  1 forcing (control, OOS window).
#
# Same harness/guardrails as run-inference-4k-response.sh; the only difference
# is the OOS config (2015-01-01 -> 2021-12-16), whose saved
# time_mean_diagnostics.nc is the out-of-sample time-mean used for the gen -
# target near-surface-T bias maps.
#
# Usage (run FROM this configs directory):
#   ./run-inference-oos-spatial.sh                 # launch both jobs
#   ./run-inference-oos-spatial.sh noco2           # substring filter
set -euo pipefail

WANDB_IDENTITY="mcgibbon"
SCRIPT_PATH=$(git rev-parse --show-prefix)
REPO_ROOT=$(git rev-parse --show-toplevel)
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')

WANDB_USERNAME=${WANDB_USERNAME:-$WANDB_IDENTITY}
if [[ "$WANDB_USERNAME" != "$WANDB_IDENTITY" ]]; then
  echo "ERROR: WANDB_USERNAME='$WANDB_USERNAME' but runs must attribute to '$WANDB_IDENTITY'." >&2
  echo "       (BEAKER_USERNAME='$BEAKER_USERNAME' would misattribute to the wandb service account.)" >&2
  exit 1
fi
if [[ -z "$SCRIPT_PATH" ]]; then
  echo "ERROR: run this script FROM its own configs directory, not the repo root." >&2
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

JOB_GROUP="ace2s-shield-amip-only-v2-oos-spatial"
CKPT_CO2="01KX2KCETE73NA1F4BXNVDKJAP"    # with-CO2 arm   (wandb 08agor1k)
CKPT_NOCO2="01KX1X29KHM9K03MCTW0RX3CWR"  # no-CO2 arm     (wandb nogznxwv)
CONFIG_OOS="ace-inference-shield-amip-control-oos.yaml"

cd "$REPO_ROOT"
python -m fme.ace.validate_config --config_type inference "$SCRIPT_PATH/$CONFIG_OOS"

launch_job () {
  local job_name="$1"; local ckpt_dataset="$2"; local config_filename="$3"
  local config_path="$SCRIPT_PATH/$config_filename"
  should_run "$config_filename" "$job_name" || { echo "skip (filter): $job_name"; return 0; }
  [[ -f "$config_path" ]] || { echo "ERROR: config not found: $REPO_ROOT/$config_path" >&2; exit 1; }
  echo "launching: $job_name  ($config_path, ckpt $ckpt_dataset)"
  cd "$REPO_ROOT" && gantry run \
    --name "$job_name" \
    --task-name "$job_name" \
    --description 'SHiELD-AMIP-only v2 prescribed-SST OOS spatial near-surface-T inference' \
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
    --dataset "${ckpt_dataset}:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar" \
    --gpus 1 \
    --shared-memory 50GiB \
    --allow-dirty \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- python -I -m fme.ace.inference "$config_path"
}

launch_job "${JOB_GROUP}-co2-control"   "$CKPT_CO2"   "$CONFIG_OOS"
launch_job "${JOB_GROUP}-noco2-control" "$CKPT_NOCO2" "$CONFIG_OOS"
