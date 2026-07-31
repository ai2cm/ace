#!/bin/bash
# Real-hardware preemption test for PR #1398 (graceful SIGTERM teardown).
#
# Trains the ERA5 baseline config on 4 GPUs with an artificial delay injected
# into the restart-checkpoint write on SIGTERM (see the commit on this branch
# touching fme/core/generics/trainer.py). Let the job train for a few hundred
# batches, then preempt/stop it in beaker and check the logs:
#
#   - "Received SIGTERM, shutting down the distributed backend" on every rank
#   - "Shutting down rank N" on every rank, within ~1s of the signal
#   - "SIMULATION: delaying restart checkpoint write by ..." on rank 0
#   - with the default 45s delay: exactly one "forcefully exiting via 9"
#     (rank 0, mid-write) and NO "Invalid access of peer GPU memory over
#     nvlink" anywhere; with 15s: no force-kill at all and
#     "SIMULATION: delay complete" + "Saving latest checkpoint" on rank 0
#   - afterwards: node shows no SXid errors and is not cordoned
#
# Must be run from the exp/simulate-slow-preemption-checkpoint branch, pushed
# to remote (gantry installs the current commit).
#
# Usage: ./run-preemption-test.sh [delay_seconds]
#   delay_seconds: default 45, which exceeds torchrun's 30s grace budget
#                  (worst case: rank 0 is killed mid-write, harmlessly).
#                  Use 15 for the realistic case where the write completes.

set -e

DELAY="${1:-45}"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=4
CONFIG_PATH="configs/baselines/era5/ace-train-config-1-step-pretrain.yaml"
JOB_NAME="preemption-teardown-test-${DELAY}s-$(date +%Y%m%d-%H%M%S)"

cd "$REPO_ROOT"  # so config path is valid no matter where we are running this script

echo "Submitting $JOB_NAME from branch $(git branch --show-current)" \
  "@ $(git rev-parse --short HEAD)"

python -m fme.ace.validate_config --config_type train "$CONFIG_PATH"

# Extract additional args from config header
extra_args=()
while IFS= read -r line; do
  [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
done < "$CONFIG_PATH"

gantry run \
  --name "$JOB_NAME" \
  --task-name "$JOB_NAME" \
  --description "Preemption teardown test for PR #1398 (simulated ${DELAY}s restart-checkpoint write)" \
  --beaker-image "$(cat "$REPO_ROOT"/latest_deps_only_image.txt)" \
  --workspace ai2/ace \
  --priority normal \
  --preemptible \
  --cluster ai2/titan \
  --env WANDB_USERNAME="$WANDB_USERNAME" \
  --env WANDB_NAME="$JOB_NAME" \
  --env WANDB_JOB_TYPE=training \
  --env WANDB_RUN_GROUP=preemption-teardown-test \
  --env FME_SIMULATED_RESTART_CHECKPOINT_DELAY="$DELAY" \
  --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset-secret google-credentials:/tmp/google_application_credentials.json \
  --gpus $N_GPUS \
  --shared-memory 400GiB \
  --weka climate-default:/climate-default \
  --budget ai2/atec-climate \
  --system-python \
  --install "pip install --no-deps ." \
  "${extra_args[@]}" \
  -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_PATH
