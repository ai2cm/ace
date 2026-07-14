#!/bin/bash
#
# Launch COUPLED slab-ocean SOM 1xCO2 evaluator jobs for all three 4deg/daily
# ACE2S-SHiELD+ arms, using eval-paperarm-som-1xco2-coupled.yaml (which adds a
# stepper_override.ocean.slab block to flip the prescribed-SST checkpoints into
# coupled slab-ocean inference). Produces time_mean_diagnostics.nc
# (gen_map / target_map / bias_map) for a like-for-like SOM bias-map comparison
# (report Fig 5) against the paper's 1-degree coupled slab-ocean row.
#
# Usage (run FROM this configs directory, not the repo root):
#   ./run-eval-som-coupled.sh              # launch all 3
#   ./run-eval-som-coupled.sh paper        # only jobs matching "paper"
set -euo pipefail

# === GUARDRAILS ===
WANDB_IDENTITY="mcgibbon"
SCRIPT_PATH=$(git rev-parse --show-prefix)
REPO_ROOT=$(git rev-parse --show-toplevel)
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-$WANDB_IDENTITY}
if [[ "$WANDB_USERNAME" != "$WANDB_IDENTITY" ]]; then
  echo "ERROR: WANDB_USERNAME='$WANDB_USERNAME' must be '$WANDB_IDENTITY'." >&2; exit 1
fi
if [[ -z "$SCRIPT_PATH" ]]; then
  echo "ERROR: run FROM the configs directory, not the repo root." >&2; exit 1
fi
# === END GUARDRAILS ===

LAUNCH_FILTERS=("$@")
should_run() {
  [[ ${#LAUNCH_FILTERS[@]} -eq 0 ]] && return 0
  local f; for f in "${LAUNCH_FILTERS[@]}"; do [[ "$1" == *"$f"* ]] && return 0; done; return 1
}

JOB_GROUP="ace2s-shield-plus-4deg-daily-som-coupled"
CFG="eval-paperarm-som-1xco2-coupled.yaml"

# best_inference_ckpt result datasets of the three done arms.
CKPT_PAPER="01KWWQGQGRCFGQMVD6E9WVWGNJ"  # paper-faithful  (wandb f3fhd6hn)
CKPT_V2="01KWQSYAEMVSYGQ4AD5NH4XK5P"     # v2 builder      (wandb 2ijn8ynx)
CKPT_V2RES="01KWYYTT4Q42Y1ME622T8JHF06"  # v2 + residual   (wandb wh6j37yg)

cd "$REPO_ROOT"
python -m fme.ace.validate_config --config_type evaluator "$SCRIPT_PATH/$CFG"

launch_job () {
  local job_name="$1" ckpt="$2"
  should_run "$job_name" || { echo "skip: $job_name"; return 0; }
  echo "launching: $job_name (ckpt $ckpt, $CFG)"
  cd "$REPO_ROOT" && gantry run \
    --name "$job_name" --task-name "$job_name" \
    --description '4deg/daily ACE2S-SHiELD+ coupled slab-ocean SOM 1xCO2 evaluator' \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority high --not-preemptible \
    --cluster ai2/jupiter --cluster ai2/titan \
    --env WANDB_USERNAME="$WANDB_USERNAME" --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP="$JOB_GROUP" \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "${ckpt}:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar" \
    --gpus 1 --shared-memory 50GiB --allow-dirty \
    --weka climate-default:/climate-default --budget ai2/atec-climate \
    --system-python --install "pip install --no-deps ." \
    -- python -I -m fme.ace.evaluator "$SCRIPT_PATH/$CFG"
}

launch_job "${JOB_GROUP}-paper"  "$CKPT_PAPER"
launch_job "${JOB_GROUP}-v2"     "$CKPT_V2"
launch_job "${JOB_GROUP}-v2res"  "$CKPT_V2RES"
