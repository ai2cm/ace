#!/bin/bash
#
# Launch the 1-degree ERA5 hybrid-residual training experiment.
#
# Arm 1: every prognostic stepped as a residual-normalized tendency except
# specific_total_water_0 (full-field); diagnostics are full-field by
# construction. Everything else is byte-identical to the ERA5 baseline
# 1-step pretrain (configs/baselines/era5/ace-train-config-1-step-pretrain.yaml),
# so the existing ace2s baseline run is the full-field control.
#
# Requires the residual_prediction_names feature (PR #1487); launch from a
# branch that contains it.
#
# Examples:
#   ./launch-hybridresid.sh
#   DRY_RUN=1 ./launch-hybridresid.sh

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
# Atmosphere-only training fits H100s; jupiter with 8 GPUs avoids the titan
# queue (Troy, 2026-09-11).
N_GPUS="${N_GPUS:-8}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
PRIORITY="${PRIORITY:-normal}"
JOB_GROUP="${JOB_GROUP:-era5-1deg-hybrid-residual}"

REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#"$REPO_ROOT"/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}

cd "$REPO_ROOT"

launch() {
  local config_filename="$1"
  local job_name="$2"
  local config="${SCRIPT_PATH}/${config_filename}"

  python -m fme.ace.validate_config --config_type train "$config"

  # Extract additional args (e.g. stats dataset mounts) from the config header.
  local extra_args=()
  while IFS= read -r line; do
    [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
  done < "$config"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [dry-run] $job_name ($config) extra args: ${extra_args[*]}"
    return
  fi

  gantry run \
    --name "$job_name" \
    --task-name "$job_name" \
    --description "ERA5 1deg hybrid-residual training (${config_filename})" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace \
    --priority "$PRIORITY" \
    --min-runtime "${MIN_RUNTIME:-8h}" \
    --cluster "$CLUSTER" \
    --weka climate-default:/climate-default \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --env FME_COLLECTIVE_TIMEOUT_MINUTES=120 \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP="$JOB_GROUP" \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    "${extra_args[@]}" \
    --gpus "$N_GPUS" \
    --shared-memory 400GiB \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.ace.train "$config"
}

launch "hybridresid-train.yaml" "era5-1deg-hybridresid-rs0"
