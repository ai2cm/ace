#!/bin/bash
#
# Launch the 6 prescribed-SST inference jobs for the SHiELD-AMIP-only +4K
# land-amplification response evaluation:
#   2 checkpoints (with-CO2 / no-CO2)  x  3 forcings (control / p2k / p4k).
#
# The +2K/+4K warming comes entirely from pointing the forcing loader at the
# p2k/p4k target datasets (SST prescribed over ocean via the checkpoint's
# ocean interpolate config) — there is no perturbations block. Normalization
# and all variable lists come from the mounted checkpoint, so the same forcing
# config serves both checkpoints.
#
# Usage (run FROM this configs directory, not the repo root):
#   ./run-inference-4k-response.sh                 # launch all 6 jobs
#   ./run-inference-4k-response.sh p4k             # only jobs whose config or
#                                                  #   job name matches "p4k"
#   ./run-inference-4k-response.sh co2-control     # OR-match multiple substrings
set -euo pipefail

# === GUARDRAILS (adapted from research/.../launching-runs/run-train.reference.sh) ===
WANDB_IDENTITY="mcgibbon"   # the wandb username every run must attribute to

SCRIPT_PATH=$(git rev-parse --show-prefix)   # repo-root-relative dir of this script
REPO_ROOT=$(git rev-parse --show-toplevel)
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')

# WANDB attribution guard. The beaker job env does not carry WANDB_USERNAME, and
# the beaker account (jeremym) makes wandb fall back to the API-key service
# account, so an unset/null/jeremym value silently misattributes the run. Beaker
# specs are immutable, so a miss costs a full stop+relaunch+rewrite-every-record
# cycle — fail loud, before submit.
WANDB_USERNAME=${WANDB_USERNAME:-$WANDB_IDENTITY}
if [[ "$WANDB_USERNAME" != "$WANDB_IDENTITY" ]]; then
  echo "ERROR: WANDB_USERNAME='$WANDB_USERNAME' but runs must attribute to '$WANDB_IDENTITY'." >&2
  echo "       (BEAKER_USERNAME='$BEAKER_USERNAME' would misattribute to the wandb service account.)" >&2
  echo "       Run:  export WANDB_USERNAME=$WANDB_IDENTITY   before launching." >&2
  exit 1
fi

# cwd / path guard: an empty SCRIPT_PATH means the script was run from the repo
# root, which would produce a bad "/<config>.yaml" path and submit doomed jobs.
if [[ -z "$SCRIPT_PATH" ]]; then
  echo "ERROR: SCRIPT_PATH (git rev-parse --show-prefix) is empty." >&2
  echo "       Invoke this script FROM its own configs directory, not the repo root." >&2
  exit 1
fi

LAUNCH_FILTERS=("$@")
should_run() {  # should_run <config_filename> <job_name>
  [[ ${#LAUNCH_FILTERS[@]} -eq 0 ]] && return 0
  local f
  for f in "${LAUNCH_FILTERS[@]}"; do
    [[ "$1" == *"$f"* || "$2" == *"$f"* ]] && return 0
  done
  return 1
}
# === END GUARDRAILS =========================================================

JOB_GROUP="ace2s-shield-amip-only-v2-4k-response"

# Best-inference checkpoints of the two DONE training runs (result datasets).
CKPT_CO2="01KX2KCETE73NA1F4BXNVDKJAP"    # with-CO2 arm   (wandb 08agor1k)
CKPT_NOCO2="01KX1X29KHM9K03MCTW0RX3CWR"  # no-CO2 arm     (wandb nogznxwv)

CONFIG_CONTROL="ace-inference-shield-amip-control.yaml"
CONFIG_P2K="ace-inference-shield-amip-p2k.yaml"
CONFIG_P4K="ace-inference-shield-amip-p4k.yaml"

cd "$REPO_ROOT"

# Validate every config once, up front, before paying for any GPU spin-up.
for cfg in "$CONFIG_CONTROL" "$CONFIG_P2K" "$CONFIG_P4K"; do
  python -m fme.ace.validate_config --config_type inference "$SCRIPT_PATH/$cfg"
done

launch_job () {
  local job_name="$1"
  local ckpt_dataset="$2"
  local config_filename="$3"
  local config_path="$SCRIPT_PATH/$config_filename"

  should_run "$config_filename" "$job_name" || { echo "skip (filter): $job_name"; return 0; }

  if [[ ! -f "$config_path" ]]; then
    echo "ERROR: config not found: $REPO_ROOT/$config_path" >&2
    exit 1
  fi

  echo "launching: $job_name  ($config_path, ckpt $ckpt_dataset)"

  cd "$REPO_ROOT" && gantry run \
    --name "$job_name" \
    --task-name "$job_name" \
    --description 'SHiELD-AMIP-only v2 prescribed-SST +4K response inference' \
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

# 2 checkpoints x 3 forcings = 6 jobs.
launch_job "${JOB_GROUP}-co2-control"   "$CKPT_CO2"   "$CONFIG_CONTROL"
launch_job "${JOB_GROUP}-co2-p2k"       "$CKPT_CO2"   "$CONFIG_P2K"
launch_job "${JOB_GROUP}-co2-p4k"       "$CKPT_CO2"   "$CONFIG_P4K"
launch_job "${JOB_GROUP}-noco2-control" "$CKPT_NOCO2" "$CONFIG_CONTROL"
launch_job "${JOB_GROUP}-noco2-p2k"     "$CKPT_NOCO2" "$CONFIG_P2K"
launch_job "${JOB_GROUP}-noco2-p4k"     "$CKPT_NOCO2" "$CONFIG_P4K"
