
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
    --workspace ai2/climate-titan \
    --priority urgent \
    --cluster ai2/titan \
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

# run_training "train-4deg-daily-era5-only-co2.yaml" "train-4deg-daily-era5-only-co2-rs0" 1
# run_training "train-4deg-daily-era5-only-fpgm.yaml" "train-4deg-daily-era5-only-rs0-fpgm" 1
# run_training "train-4deg-daily-era5-only.yaml" "train-4deg-daily-era5-only-rs0" 1
# run_training "train-4deg-daily-era5-only-tnorm.yaml" "train-4deg-daily-era5-only-rs0-tnorm" 1
# run_training "train-4deg-daily-era5-only-local-mlp.yaml" "train-4deg-daily-era5-only-local-mlp" 1
# run_training "train-4deg-daily-era5-only-local-mlp-diagnostics.yaml" "train-4deg-daily-era5-only-local-mlp-diagnostics" 1
# run_training "train-4deg-daily-era5-only-ankur-local-mlp-diagnostics.yaml" "train-4deg-daily-era5-only-ankur-local-mlp-diagnostics" 1
# run_training "train-4deg-daily-era5-only-local-mlp.yaml" "train-4deg-daily-era5-only-local-mlp-rs0" 1
# run_training "train-4deg-daily-labels-co2.yaml" "train-4deg-daily-labels-co2-rs0" 1
# run_training "train-4deg-6hourly-era5-only-co2.yaml" "train-4deg-6hourly-era5-only-co2-rs0" 1
# run_training "train-1deg-6hourly-era5-only-co2.yaml" "train-1deg-6hourly-era5-only-co2-rs0" 4
# run_training "train-1deg-daily-era5-only-co2.yaml" "train-1deg-daily-era5-only-co2-rs0" 4
# run_training "train-1deg-daily-era5-only-tnorm.yaml" "train-1deg-daily-era5-only-rs0-tnorm" 4
# run_training "train-4deg-6hourly-era5-only.yaml" "train-4deg-6hourly-era5-only-rs0" 1
# run_training "train-1deg-6hourly-era5-only.yaml" "train-1deg-6hourly-era5-only-rs0" 4
# run_training "train-1deg-daily-era5-only.yaml" "train-1deg-daily-era5-only-rs0" 4
# run_training "train-4deg-daily-era5-only-shared-t.yaml" "train-4deg-daily-era5-only-rs0-shared-t" 1
# run_training "train-4deg-daily-era5-only-shared-t-append.yaml" "train-4deg-daily-era5-only-rs0-shared-t-append" 1
# run_training "train-4deg-daily-era5-only-perchan.yaml" "train-4deg-daily-era5-only-rs0-perchan" 1
# run_training "train-4deg-daily-era5-only-perchan-append.yaml" "train-4deg-daily-era5-only-rs0-perchan-append" 1
# run_training "train-4deg-daily-era5-only-rlgm.yaml" "train-4deg-daily-era5-only-rs0-rlgm" 1
# run_training "train-4deg-daily-era5-only-shared-t-append-rlgm.yaml" "train-4deg-daily-era5-only-rs0-shared-t-append-rlgm" 1
# run_training "train-4deg-daily-era5-only-rlgm-cf.yaml" "train-4deg-daily-era5-only-rs0-rlgm-cf" 1
# run_training "train-4deg-daily-era5-only-rlgm-ce.yaml" "train-4deg-daily-era5-only-rs0-rlgm-ce" 1
# run_training "train-4deg-daily-era5-only-rlgm-ao.yaml" "train-4deg-daily-era5-only-rs0-rlgm-ao" 1
# run_training "train-4deg-daily-era5-only-rlgm-cf-ao.yaml" "train-4deg-daily-era5-only-rs0-rlgm-cf-ao" 1
# run_training "train-4deg-daily-era5-only-rlgm-ce-ao.yaml" "train-4deg-daily-era5-only-rs0-rlgm-ce-ao" 1
# run_training "train-4deg-daily-era5-only-rlgm-ce-ao-n05.yaml" "train-4deg-daily-era5-only-rs0-rlgm-ce-ao-n05" 1
# run_training "train-4deg-daily-era5-only-shared-t-append-rlgm-ce-ao.yaml" "train-4deg-daily-era5-only-rs0-shared-t-append-rlgm-ce-ao" 1
# run_training "train-4deg-daily-era5-only-shared-t-append-rlgm-ce.yaml" "train-4deg-daily-era5-only-rs0-shared-t-append-rlgm-ce" 1
# run_training "train-4deg-daily-era5-only-shared-t-append-rlgm-ao.yaml" "train-4deg-daily-era5-only-rs0-shared-t-append-rlgm-ao" 1
# run_training "train-4deg-daily-era5-only-shared-t-append-rlgm-ce-ao-n05.yaml" "train-4deg-daily-era5-only-rs0-shared-t-append-rlgm-ce-ao-n05" 1
# run_training "train-4deg-daily-era5-only-shared-t-append-rlgm-ce-n05.yaml" "train-4deg-daily-era5-only-rs0-shared-t-append-rlgm-ce-n05" 1
run_training "train-4deg-daily-era5-only-shared-t-append-n1k.yaml" "train-4deg-daily-era5-only-rs0-shared-t-append-n1k" 1
run_training "train-4deg-daily-era5-only-shared-t-append-n2k.yaml" "train-4deg-daily-era5-only-rs0-shared-t-append-n2k" 1
# run_training "train-1deg-6hourly-era5-only-shared-t.yaml" "train-1deg-6hourly-era5-only-rs0-shared-t" 4
# run_training "train-1deg-6hourly-era5-only-shared-t-append.yaml" "train-1deg-6hourly-era5-only-rs0-shared-t-append" 4
# run_training "train-1deg-daily-era5-only-shared-t.yaml" "train-1deg-daily-era5-only-rs0-shared-t" 4
# run_training "train-1deg-daily-era5-only-shared-t-append.yaml" "train-1deg-daily-era5-only-rs0-shared-t-append" 4
