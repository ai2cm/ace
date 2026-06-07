
#!/bin/bash

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
REPO_ROOT=$(git rev-parse --show-toplevel)

cd "$REPO_ROOT"

run_training() {
  local config_filename="$1"
  local job_name="$2"
  local N_GPUS="$3"
  local WORKSPACE="${4:-ai2/climate-titan}"
  local PRIORITY="${5:-urgent}"
  local CLUSTER="${6:-ai2/titan}"
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"

  python -m fme.ace.validate_config --config_type train "$CONFIG_PATH"

  # Extract additional args from config header
  local extra_args=()
  while IFS= read -r line; do
    [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
  done < "$CONFIG_PATH"

  gantry run \
    --name "$job_name" \
    --description 'Run ACE training (AIMIP-like baseline)' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace "$WORKSPACE" \
    --priority "$PRIORITY" \
    --cluster "$CLUSTER" \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP= \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --gpus "$N_GPUS" \
    --shared-memory 400GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --allow-dirty \
    --system-python \
    --install "pip install --no-deps ." \
    "${extra_args[@]}" \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.ace.train "$CONFIG_PATH"
}

# --- Wave 1 (climate-titan, urgent) ---
# run_training "train-4deg-daily-v1-era5-only.yaml" "train-4deg-daily-v1-era5-only-rs0" 1
# run_training "train-4deg-daily-v1-labels.yaml" "train-4deg-daily-v1-labels-rs0" 1
# run_training "train-4deg-daily-v1-era5-only-residual.yaml" "train-4deg-daily-v1-era5-only-residual-rs0" 1
# run_training "train-4deg-daily-v1-era5-only-lr-tuning.yaml" "train-4deg-daily-v1-era5-only-lr-tuning-rs0" 1

# --- Wave 2 (Jupiter, high) --- [canceled by user]
# run_training "train-4deg-daily-v1-labels-384-lr-tuning.yaml" "train-4deg-daily-v1-labels-384-lr-tuning-rs0" 1 ai2/ace high ai2/jupiter
# run_training "train-4deg-daily-v1-labels-384-residual-lr-tuning.yaml" "train-4deg-daily-v1-labels-384-residual-lr-tuning-rs0" 1 ai2/ace high ai2/jupiter
# run_training "train-4deg-daily-v1-era5-only-384-residual-lr-tuning.yaml" "train-4deg-daily-v1-era5-only-384-residual-lr-tuning-rs0" 1 ai2/ace high ai2/jupiter
# run_training "train-4deg-daily-v1-era5-only-rs1.yaml" "train-4deg-daily-v1-era5-only-rs1" 1 ai2/ace high ai2/jupiter
# run_training "train-4deg-daily-v1-era5-only-lr-tuning-rs1.yaml" "train-4deg-daily-v1-era5-only-lr-tuning-rs1" 1 ai2/ace high ai2/jupiter
# run_training "train-4deg-daily-v1-labels-residual-lr-tuning-rs1.yaml" "train-4deg-daily-v1-labels-residual-lr-tuning-rs1" 1 ai2/ace high ai2/jupiter
# run_training "train-4deg-daily-v1-era5-only-256-lr-tuning.yaml" "train-4deg-daily-v1-era5-only-256-lr-tuning-rs0" 1 ai2/ace high ai2/jupiter

# --- Wave 3: Residual drift fixes (Jupiter, high) ---
# run_training "train-4deg-daily-v1-era5-only-residual-winds-anomaly-ft.yaml" "train-4deg-daily-v1-era5-only-residual-winds-anomaly-ft-rs0" 1 ai2/ace high ai2/jupiter  # already running
# run_training "train-4deg-daily-v1-era5-only-residual-all-anomaly-ft.yaml" "train-4deg-daily-v1-era5-only-residual-all-anomaly-ft-rs0" 1 ai2/ace high ai2/jupiter  # already running
# --- Wave 3b: tend-reg relaunch after gradient-accumulation backward fix (21f54000f) --- [finished]
# run_training "train-4deg-daily-v1-era5-only-residual-tend-reg-ft.yaml" "train-4deg-daily-v1-era5-only-residual-tend-reg-ft-rs0" 1 ai2/ace high ai2/jupiter
# run_training "train-4deg-daily-v1-era5-only-residual-winds-anomaly-tend-reg-ft.yaml" "train-4deg-daily-v1-era5-only-residual-winds-anomaly-tend-reg-ft-rs0" 1 ai2/ace high ai2/jupiter

# --- Wave 4: label-conditioned (ERA5 + c96-shield) residual, tend-reg weight 0.05, from-scratch 60ep (Jupiter, high) ---
run_training "train-4deg-daily-v1-labels-residual-winds-anomaly-tend-reg.yaml" "train-4deg-daily-v1-labels-residual-winds-anomaly-tend-reg-rs0" 1 ai2/ace high ai2/jupiter
run_training "train-4deg-daily-v1-labels-residual-winds-anomaly-tend-reg-multistep.yaml" "train-4deg-daily-v1-labels-residual-winds-anomaly-tend-reg-multistep-rs0" 1 ai2/ace high ai2/jupiter
run_training "train-4deg-daily-v1-labels-residual-all-anomaly-tend-reg-multistep.yaml" "train-4deg-daily-v1-labels-residual-all-anomaly-tend-reg-multistep-rs0" 1 ai2/ace high ai2/jupiter
run_training "train-4deg-daily-v1-labels-residual-tend-reg.yaml" "train-4deg-daily-v1-labels-residual-tend-reg-rs0" 1 ai2/ace high ai2/jupiter
