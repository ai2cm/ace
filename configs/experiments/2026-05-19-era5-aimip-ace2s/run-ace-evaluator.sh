#!/bin/bash

set -e
export GRPC_VERBOSITY=ERROR

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
WANDB_USERNAME=bhenn1983
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

CONFIG_PATH="$SCRIPT_PATH/ace-evaluator-config.yaml"
STATS_DATASET="andrep/2026-03-19-era5-1deg-8layer-stats-1990-2019"

python -m fme.ace.validate_config --config_type evaluator "$CONFIG_PATH"

run_evaluation() {
  local job_name="$1"
  local checkpoint_dataset="$2"

  gantry run \
    --name "$job_name" \
    --task-name "$job_name" \
    --description 'Run ACE ERA5 AIMIP evaluation' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority high \
    --not-preemptible \
    --cluster ai2/ceres-cirrascale \
    --cluster ai2/titan-cirrascale \
    --cluster ai2/saturn-cirrascale \
    --cluster ai2/jupiter-cirrascale-2 \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=inference \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "$STATS_DATASET":/statsdata \
    --dataset "$checkpoint_dataset":training_checkpoints/best_inference_ckpt.tar:/ckpt.tar \
    --gpus 1 \
    --shared-memory 200GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- python -m fme.ace.evaluator "$CONFIG_PATH"
}

# Seed name → beaker checkpoint dataset ID from the completed PLFT job.
# Names drop the "-pressure-level-fine-tuning" suffix since all seeds have gone through PLFT.
# Fill in TBD IDs once stochastic PLFT jobs complete.

# deterministic seeds
run_evaluation "ace2-1-aimip-era5-rs0-evaluator"                           "01KT0PQ3W374JX82TGM41TPJXD"
run_evaluation "ace2-1-aimip-era5-rs1-evaluator"                           "01KT9Q70V5CDJ5Y9T1JR2BRGVB"
run_evaluation "ace2-1-aimip-era5-energy-corrector-rs0-evaluator"          "01KSYDZ80YW63MQ0QN0DR5F4VD"
run_evaluation "ace2-1-aimip-era5-energy-corrector-rs1-evaluator"          "01KT9Q6PX1FF7TZ2N4VD50EMCQ"
run_evaluation "ace2-1-aimip-era5-energy-corrector-unacc5p7-rs0-evaluator" "01KT8JRN0D25D8NSDCKY8SEMHM"

# stochastic seeds (no rs1); uncomment and fill in IDs once PLFT jobs complete
run_evaluation "ace2s-aimip-era5-rs0-evaluator"                 "TBD"
run_evaluation "ace2s-aimip-era5-energy-corrector-rs0-evaluator" "TBD"
