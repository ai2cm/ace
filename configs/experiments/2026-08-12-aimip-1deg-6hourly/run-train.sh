#!/bin/bash

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
REPO_ROOT=$(git rev-parse --show-toplevel)

# wandb runs on a service-account key, so this is the only thing attributing them to a
# human. It differs from the beaker username, so it cannot be derived -- fail before submit.
WANDB_IDENTITY="bhenn1983"
WANDB_USERNAME=${WANDB_USERNAME:-$WANDB_IDENTITY}
if [[ "$WANDB_USERNAME" != "$WANDB_IDENTITY" ]]; then
  echo "ERROR: WANDB_USERNAME='$WANDB_USERNAME' but runs must attribute to '$WANDB_IDENTITY'." >&2
  echo "       Unset WANDB_USERNAME, or export WANDB_USERNAME=$WANDB_IDENTITY, before launching." >&2
  exit 1
fi

# Single cluster on purpose: per-GPU memory is cluster-specific, so N_GPUS is only correct
# for the cluster below. Retarget by changing both together, not by adding a --cluster.
N_GPUS=4

# cwd guard: an empty SCRIPT_PATH means this was run from the repo root, which would make
# CONFIG_PATH absolute and submit a doomed job.
if [[ -z "$SCRIPT_PATH" ]]; then
  echo "ERROR: run this script from configs/experiments/2026-08-12-aimip-1deg-6hourly, not the repo root." >&2
  exit 1
fi

cd "$REPO_ROOT"  # so the config path is valid no matter where this is run from

run_training() {
  local config_filename="$1"
  local job_name="$2"
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"

  python -m fme.ace.validate_config --config_type train "$CONFIG_PATH"

  # Extract additional args from config header
  local extra_args=()
  while IFS= read -r line; do
    [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
  done < "$CONFIG_PATH"

  gantry run \
    --name "$job_name" \
    --task-name "$job_name" \
    --description 'Run ACE training (AIMIP-like baseline, 1°/6-hourly)' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/climate-titan \
    --priority urgent \
    --timeout 0 \
    --no-logs \
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
    --system-python \
    --install "pip install --no-deps ." \
    "${extra_args[@]}" \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.ace.train "$CONFIG_PATH"
}

# 6-hourly retrain of the 1deg/daily v2 ERA5-only no-residual no-CO2 baseline submitted to
# AIMIP. The daily-step original emits daily snapshots, which cannot be evaluated against
# the monthly-/daily-AVERAGED AIMIP ERA5 data; a 6h step resolves the diurnal cycle so its
# output averages into comparable means. See README.md for the full change list.
#
# Trained in three stages: 1-step pretraining, a multi-step fine-tune initialized from its
# checkpoint, then a pressure-level fine-tune adding the AIMIP plev diagnostics. Each stage's
# config already carries our run's donor dataset id; to reproduce from scratch, substitute the
# id your own preceding stage produced.

base_name="train-1deg-6hourly-v2-era5-only-no-residual-no-co2"

# --- Stage 1: 1-step pretraining ---
run_training "$base_name.yaml" "$base_name-rs0"

# --- Stage 2: multi-step fine-tuning ---   [completed; see the record under stage 3]
# The `# arg:` header of $base_name-multi-step-ft.yaml mounts the stage 1 result dataset at
# /weights, supplying both the stepper config and the initial weights. Uncomment below to run.
#
# The stage 1 run this experiment's fine-tune config currently points at:
#   beaker experiment 01KZYJ4HT4ZMZH296KBNWMPCQF   launched 2026-08-13 at commit b4a688d85
#   wandb            g94277n6                     40 epochs, 44.5h on 4 GPUs
#   result dataset   01KZYJ4HTBWED5VG3VFTRYKDRC   <- the donor checkpoint
#
# run_training "$base_name-multi-step-ft.yaml" "$base_name-multi-step-ft-rs0"

# --- Stage 3: pressure-level fine-tuning ---
# Adds the AIMIP evaluation pressure-level diagnostics (1000/850/700/500/250/100/50 hPa for
# ta/hus/ua/va/zg) via a secondary decoder head, with the trunk frozen. The config's `# arg:`
# header already mounts the stage 2 donor; uncomment below to run. NOTE: the donor dataset must
# be committed (i.e. its job fully finalized) before beaker will mount it.
#
# The stage 2 run the donor came from (completed 2026-08-24, 40 epochs, 81h wall across 35
# preempted attempts; best_val_loss 0.16528, best_inference_error 0.026124 -- unchanged from
# epoch 8, so best_inference_ckpt.tar is the epoch-8 checkpoint):
#   beaker experiment 01M06W0WN4WY1HJBWFXJNJEXC7   workspace ai2/ace, priority high
#   wandb            78crdqjr
#   result dataset   01M0RFP2DKAGABV89KRPMXX5C3   <- the donor, already in the config header
#
# run_training "$base_name-plev-ft.yaml" "$base_name-plev-ft-rs0"
