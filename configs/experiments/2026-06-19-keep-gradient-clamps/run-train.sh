#!/bin/bash
# A/B for keep_gradient_through_clamps (atmosphere precipitation case).
# 4 jobs: {baseline, ste} x {seed 0, seed 1}, each 1-GPU 4-degree.
# Same-seed pairs start bit-identical (STE leaves forward values unchanged)
# and diverge only through the corrector clamp gradient path. The dry_fraction
# metric on PRATEsfc is logged by the inline inference evaluator.

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to repo root
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=1
RUN_GROUP="keep-gradient-clamps-2026-06-19"
STATS_DATASET="jeremym/2023-08-09-vertically-resolved-4deg-fme-ensemble-dataset-stats"

cd "$REPO_ROOT"

run_training() {
  local config_filename="$1"
  local job_name="$2"
  local seed="$3"
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"

  python -m fme.ace.validate_config --config_type train "$CONFIG_PATH"

  gantry run \
    --name "$job_name" \
    --description 'A/B keep_gradient_through_clamps (precip), 1-GPU 4deg' \
    --beaker-image "$(cat "$REPO_ROOT"/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority high \
    --preemptible \
    --cluster ai2/jupiter \
    --cluster ai2/titan \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP="$RUN_GROUP" \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "$STATS_DATASET:/statsdata" \
    --gpus "$N_GPUS" \
    --shared-memory 60GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.ace.train "$CONFIG_PATH" \
       --override "seed=$seed"
}

# {baseline, ste} x {seed 0, seed 1}
run_training "train-4deg-baseline.yaml" "kgc-baseline-4deg-rs0" 0
run_training "train-4deg-ste.yaml"      "kgc-ste-4deg-rs0"      0
run_training "train-4deg-baseline.yaml" "kgc-baseline-4deg-rs1" 1
run_training "train-4deg-ste.yaml"      "kgc-ste-4deg-rs1"      1
