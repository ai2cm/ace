#!/bin/bash
#
# Year-split coupled AR readout eval (native-cadence writer) for the joint FT checkpoint.
# Job layout template: exp/samudrace-enso-skill 2026-02-17-samudrace-enso-eval
# (one gantry job per year, 12 monthly ICs, n_coupled_steps=146).
#
# OOS year windows (see make_year_configs_ar_readout.py):
#   val         — 0251–0255  (official FT validation; default)
#   extended20  — 0231–0250  (20y strictly unseen by train/val)
#   extended40  — 0211–0250  (full pre-train window)
#
# Examples:
#   ./evaluate-coupled-nino-ar-sst.sh
#   OOS_WINDOW=extended20 ./evaluate-coupled-nino-ar-sst.sh
#   OOS_WINDOW=custom YEAR_START=236 YEAR_END=255 ./evaluate-coupled-nino-ar-sst.sh
#
# Year YAMLs under ar_readout_year_configs/ are generated (often untracked), so
# each Beaker job regenerates its config on the worker before evaluate.

set -euo pipefail

JOB_GROUP="${JOB_GROUP:-cm4-1pct-samudra-nino}"
COUPLED_RESULTS_DATASET="${COUPLED_RESULTS_DATASET:-01KY3DATM3CAEA479JQZQDPT9W}"
COUPLED_CKPT="${COUPLED_CKPT:-best_inference_ckpt}"
OOS_WINDOW="${OOS_WINDOW:-val}"

REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
CONFIG_DIR="${SCRIPT_PATH}/ar_readout_year_configs"
MAKE_SCRIPT="${SCRIPT_PATH}/make_year_configs_ar_readout.py"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
N_GPUS=1

case "$OOS_WINDOW" in
  val)         YEAR_START=251; YEAR_END=255 ;;
  extended20) YEAR_START=231; YEAR_END=250 ;;
  extended40) YEAR_START=211; YEAR_END=250 ;;
  custom)
    : "${YEAR_START:?Set YEAR_START when OOS_WINDOW=custom}"
    : "${YEAR_END:?Set YEAR_END when OOS_WINDOW=custom}"
    ;;
  *)
    echo "Unknown OOS_WINDOW=$OOS_WINDOW (val|extended20|extended40|custom)" >&2
    exit 1
    ;;
esac

cd "$REPO_ROOT"

if [[ "$OOS_WINDOW" == "custom" ]]; then
  python "$MAKE_SCRIPT" --year-start "$YEAR_START" --year-end "$YEAR_END"
else
  python "$MAKE_SCRIPT" --oos-window "$OOS_WINDOW"
fi

echo "AR SST eval: years $(printf '%04d' "$YEAR_START")–$(printf '%04d' "$YEAR_END") (OOS_WINDOW=$OOS_WINDOW, $((YEAR_END - YEAR_START + 1)) jobs × 12 ICs)"

run_eval() {
  local year="$1"
  local year_str
  year_str=$(printf "%04d" "$year")
  local config_path="${CONFIG_DIR}/evaluator-config-1pct-ar-readout-yr${year_str}.yaml"
  local job_name="cm4-coupled-ft-nino-ar-readout-yr${year_str}"

  python -m fme.coupled.validate_config --config_type evaluator "$config_path"

  gantry run \
    --name "$job_name" \
    --task-name "$job_name" \
    --description "Coupled FT AR readout eval: native sst+nino34_lead (5d) + wind stress (6h)" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/climate-titan \
    --priority high \
    --preemptible \
    --cluster ai2/titan \
    --weka climate-default:/climate-default \
    --env WANDB_USERNAME="$BEAKER_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=evaluation \
    --env WANDB_RUN_GROUP="$JOB_GROUP" \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "$COUPLED_RESULTS_DATASET:training_checkpoints/${COUPLED_CKPT}.tar:/ckpt.tar" \
    --gpus "$N_GPUS" \
    --shared-memory 400GiB \
    --budget ai2/atec-climate \
    --allow-dirty \
    --system-python \
    --install "pip install --no-deps ." \
    -- bash -c \
        "python '$MAKE_SCRIPT' --year-start $year --year-end $year && \
         python -I -m fme.coupled.evaluator '$config_path'"
}

for year in $(seq "$YEAR_START" "$YEAR_END"); do
  run_eval "$year"
done
