#!/bin/bash
#
# Launch the raw-daily-output prescribed-SST rollouts for the SHiELD replication
# of the im4ecamc 2015-out-of-sample process diagnosis (reports PR 102):
#   checkpoints  x  {2015-2021 out-of-sample, 1995-2001 in-sample control}.
# Checkpoints: the residual no-CO2 arm (wandb nogznxwv) and the non-residual
# no-CO2 twin of im4ecamc (train-4deg-daily-v2-shield-amip-only-no-residual-no-co2).
# The twin's checkpoint dataset is filled in once its training finishes; the
# launcher refuses to submit while it is a placeholder.
#
# Usage (run FROM this configs directory):
#   ./run-inference-raw.sh                  # launch every job
#   ./run-inference-raw.sh noresidual       # substring filter on job name/config
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

JOB_GROUP="ace2s-shield-amip-only-v2-raw"
CKPT_NOCO2="01KX1X29KHM9K03MCTW0RX3CWR"            # residual no-CO2 arm (wandb nogznxwv)
CKPT_NORES_NOCO2="PLACEHOLDER_NORES_NOCO2_RESULT_DATASET"  # non-residual twin, set when trained

cd "$REPO_ROOT"
for c in ace-inference-shield-amip-raw-2015.yaml ace-inference-shield-amip-raw-1995.yaml; do
  python -m fme.ace.validate_config --config_type inference "$SCRIPT_PATH/$c"
done

launch_job () {
  local job_name="$1"; local ckpt_dataset="$2"; local config_filename="$3"
  local config_path="$SCRIPT_PATH/$config_filename"
  should_run "$config_filename" "$job_name" || { echo "skip (filter): $job_name"; return 0; }
  [[ -f "$config_path" ]] || { echo "ERROR: config not found: $REPO_ROOT/$config_path" >&2; exit 1; }
  if [[ "$ckpt_dataset" == PLACEHOLDER* ]]; then
    echo "ERROR: $job_name: checkpoint dataset is a placeholder ($ckpt_dataset)" >&2; exit 1
  fi
  echo "launching: $job_name  ($config_path, ckpt $ckpt_dataset)"
  cd "$REPO_ROOT" && gantry run \
    --name "$job_name" \
    --task-name "$job_name" \
    --description 'SHiELD-AMIP-only v2 prescribed-SST raw-output process-diagnosis inference' \
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

for y in 2015 1995; do
  launch_job "${JOB_GROUP}-noco2-${y}"            "$CKPT_NOCO2"       "ace-inference-shield-amip-raw-${y}.yaml"
  launch_job "${JOB_GROUP}-noresidual-noco2-${y}" "$CKPT_NORES_NOCO2" "ace-inference-shield-amip-raw-${y}.yaml"
done
