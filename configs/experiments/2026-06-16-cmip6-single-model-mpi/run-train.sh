#!/bin/bash
#
# Submit ACE training jobs for the single-model CMIP6 smoke test
# (MPI-ESM1-2-LR historical 1979-2014, single member r1i1p1f1). Two runs:
#   - ...-all     : every daily variable MPI publishes, incl. monthly-cadence
#                   (amon/omon/simon) + spiky/masked daily diagnostics.
#   - ...-lowrisk : drops the monthly-cadence vars and the spiky/masked daily
#                   diagnostics (oday_omldamax, oday_tossq, siday_sithick).
# Architecture from the v2 4deg-daily ERA5 baseline; data layer from the
# cmip6 multi-model run-1. See the configs' header comments for details.
#
# Pattern lifted from configs/experiments/2026-06-01-cmip6-daily-multimodel/
# run-train.sh: each run_training validates the config, picks ``# arg: …``
# headers out of the YAML for extra gantry flags, and submits a torchrun job.
# Must be invoked from this directory (SCRIPT_PATH = git rev-parse --show-prefix).

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix) # relative to the root of the repository
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
# Attribute wandb to mcgibbon explicitly: the beaker job env doesn't carry
# WANDB_USERNAME (null -> unattributed), and BEAKER_USERNAME (jeremym)
# misattributes to the service account.
WANDB_USERNAME=${WANDB_USERNAME:-mcgibbon}
REPO_ROOT=$(git rev-parse --show-toplevel)

cd "$REPO_ROOT"

run_training() {
  local config_filename="$1"
  local job_name="$2"
  local N_GPUS="$3"
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"

  # Extract additional args from config header (e.g. ``# arg: --gpus 1``).
  local extra_args=()
  while IFS= read -r line; do
    [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
  done <"$CONFIG_PATH"

  # validate_config runs inside the gantry container as the first step so
  # Weka (mounted below) is available for the data-side validators, and the
  # job fails fast on config bugs before paying for GPU spin-up.
  gantry run \
    --name "$job_name" \
    --description 'Run ACE training (single-model CMIP6 smoke test: MPI-ESM1-2-LR)' \
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

# Runs (comment out everything except the current launch target, per the
# run-1 convention, so re-running this script doesn't duplicate live jobs).
# -all and -lowrisk launched 2026-06-24 (256b11433); -no-residual added after.
# run_training "train-4deg-daily-cmip6-mpi-single-all.yaml" "train-4deg-daily-cmip6-mpi-single-all-rs0" 1
# run_training "train-4deg-daily-cmip6-mpi-single-lowrisk.yaml" "train-4deg-daily-cmip6-mpi-single-lowrisk-rs0" 1
# run_training "train-4deg-daily-cmip6-mpi-single-lowrisk-no-residual.yaml" "train-4deg-daily-cmip6-mpi-single-lowrisk-no-residual-rs0" 1
# --- v3 masked / thickness / residual-off runs (D): reads GCS v3 zarr directly
#     (engine: zarr, no Weka staging), mask_loss on, thickness predicted (zg
#     diagnosed). Uncomment the target to launch; leave the other commented so a
#     re-run doesn't duplicate a live job.
run_training "train-4deg-daily-cmip6-mpi-single-v3-masked-thickness.yaml" "train-4deg-daily-cmip6-mpi-single-v3-masked-thickness-rs0" 1
# run_training "train-4deg-daily-cmip6-mpi-single-v3-masked-thickness-no-energy.yaml" "train-4deg-daily-cmip6-mpi-single-v3-masked-thickness-no-energy-rs0" 1
