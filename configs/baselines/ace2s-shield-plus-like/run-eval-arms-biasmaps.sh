#!/bin/bash
#
# Launch the bias-map evaluator jobs for the OTHER two 4deg/daily arms (v2
# builder, v2+residual), reusing the checkpoint-agnostic paper-arm configs
# (checkpoint mounted at /ckpt.tar). Two cases per arm — AMIP 2012-2020 and SOM
# 1xCO2 — so the report's Figs 4-5 can show all three arms beside the paper's 1
# maps. The paper arm itself was launched by run-eval-paperarm-biasmaps.sh.
#
# Usage (run FROM this configs directory):
#   ./run-eval-arms-biasmaps.sh            # launch all 4
#   ./run-eval-arms-biasmaps.sh v2res      # filter by substring
set -euo pipefail

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

LAUNCH_FILTERS=("$@")
should_run() {
  [[ ${#LAUNCH_FILTERS[@]} -eq 0 ]] && return 0
  local f; for f in "${LAUNCH_FILTERS[@]}"; do [[ "$1" == *"$f"* ]] && return 0; done; return 1
}

JOB_GROUP="ace2s-shield-plus-4deg-daily-arms-biasmaps"
# best_inference_ckpt result datasets of the two done arms.
CKPT_V2="01KWQSYAEMVSYGQ4AD5NH4XK5P"     # v2 builder      (wandb 2ijn8ynx)
CKPT_V2RES="01KWYYTT4Q42Y1ME622T8JHF06"  # v2 + residual   (wandb wh6j37yg)

cd "$REPO_ROOT"
for cfg in eval-paperarm-amip-test.yaml eval-paperarm-som-1xco2.yaml; do
  python -m fme.ace.validate_config --config_type evaluator "$SCRIPT_PATH/$cfg"
done

launch_job () {
  local job_name="$1" ckpt="$2" cfg="$3"
  should_run "$job_name" || { echo "skip: $job_name"; return 0; }
  echo "launching: $job_name (ckpt $ckpt, $cfg)"
  cd "$REPO_ROOT" && gantry run \
    --name "$job_name" --task-name "$job_name" \
    --description '4deg/daily ACE2S-SHiELD+ v2/v2res evaluator: time-mean bias maps' \
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
    -- python -I -m fme.ace.evaluator "$SCRIPT_PATH/$cfg"
}

launch_job "${JOB_GROUP}-v2-amip-test"    "$CKPT_V2"    eval-paperarm-amip-test.yaml
launch_job "${JOB_GROUP}-v2-som-1xco2"    "$CKPT_V2"    eval-paperarm-som-1xco2.yaml
launch_job "${JOB_GROUP}-v2res-amip-test" "$CKPT_V2RES" eval-paperarm-amip-test.yaml
launch_job "${JOB_GROUP}-v2res-som-1xco2" "$CKPT_V2RES" eval-paperarm-som-1xco2.yaml
