#!/bin/bash
#
# Submit ACE training jobs for the CMIP6 daily multi-model run-1 cohort
# (training_run_1.md plan, v2 data at /climate-default/2026-05-22-cmip6-
# multimodel-daily-4deg-8plev-1940-2100/v2/).
#
# Pattern lifted from configs/baselines/aimip-like/run-train.sh: each
# call to ``run_training`` validates the config, picks ``# arg: …``
# comment headers out of the YAML for extra gantry flags, and submits
# a torchrun job sized for ``n_gpus``. Add new ``run_training`` lines
# for sweeps; comment out everything except the current target before
# running.

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix) # relative to the root of the repository
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
REPO_ROOT=$(git rev-parse --show-toplevel)

cd "$REPO_ROOT"

run_training() {
  local config_filename="$1"
  local job_name="$2"
  local N_GPUS="$3"
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"

  # Extract additional args from config header
  local extra_args=()
  while IFS= read -r line; do
    [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
  done <"$CONFIG_PATH"

  # ``validate_config`` runs *inside* the gantry container as the first
  # step of the bash entrypoint so that Weka (mounted via the
  # ``--weka`` flag below) is available — the validators load
  # ``index.csv`` / ``centering.nc`` to check data-side invariants.
  # Running it before ``torchrun`` lets the job fail fast on config
  # bugs without paying for GPU spin-up; ``set -e`` propagates the
  # exit so beaker marks the experiment failed at the right step.
  gantry run \
    --name "$job_name" \
    --description 'Run ACE training (CMIP6 daily multi-model run 1)' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority high \
    --cluster ai2/jupiter \
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
    -- bash -c "set -e && python -m fme.ace.validate_config --config_type train '$CONFIG_PATH' && torchrun --nproc_per_node '$N_GPUS' -m fme.ace.train '$CONFIG_PATH'"
}

# Active runs (uncomment one at a time; sweeps go in this block).
run_training "train-4deg-daily-cmip6-multimodel-per-source-norm.yaml" "train-4deg-daily-cmip6-multimodel-per-source-norm-rs0" 1
# run_training "train-4deg-daily-cmip6-multimodel-cohort-norm.yaml" "train-4deg-daily-cmip6-multimodel-cohort-norm-rs0" 1
