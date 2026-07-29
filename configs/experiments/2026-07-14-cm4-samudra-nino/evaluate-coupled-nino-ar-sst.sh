#!/bin/bash
#
# Year-split coupled AR SST Nino eval for the joint FT checkpoint.
# Job layout template: exp/samudrace-enso-skill 2026-02-17-samudrace-enso-eval
# (one gantry job per year, 12 monthly ICs, n_coupled_steps=146).
#
# Years default to the coupled FT validation / OOS window 0251–0255
# (train starts 0256; see coupled-finetune-atmos.yaml).
#
# Year YAMLs under ar_sst_year_configs/ are generated (often untracked), so
# each Beaker job regenerates its config on the worker before evaluate.

set -euo pipefail

JOB_GROUP="${JOB_GROUP:-cm4-1pct-samudra-nino}"
COUPLED_RESULTS_DATASET="${COUPLED_RESULTS_DATASET:-01KY3DATM3CAEA479JQZQDPT9W}"
COUPLED_CKPT="${COUPLED_CKPT:-best_inference_ckpt}"
YEAR_START="${YEAR_START:-251}"
YEAR_END="${YEAR_END:-255}"

REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
CONFIG_DIR="${SCRIPT_PATH}/ar_sst_year_configs"
MAKE_SCRIPT="${SCRIPT_PATH}/make_year_configs_ar_sst.py"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
N_GPUS=1

cd "$REPO_ROOT"

run_eval() {
  local year="$1"
  local year_str
  year_str=$(printf "%04d" "$year")
  local config_path="${CONFIG_DIR}/evaluator-config-1pct-ar-sst-yr${year_str}.yaml"
  local job_name="cm4-coupled-ft-nino-ar-sst-yr${year_str}"

  # Local smoke-validate (also materializes the year YAML if missing).
  python "$MAKE_SCRIPT" --year-start "$year" --year-end "$year"
  python -m fme.coupled.validate_config --config_type evaluator "$config_path"

  gantry run \
    --name "$job_name" \
    --task-name "$job_name" \
    --description "Coupled FT AR SST Nino eval (12 monthly ICs, 146 ocean steps)" \
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
