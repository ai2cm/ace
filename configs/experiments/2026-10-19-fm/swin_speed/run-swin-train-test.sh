#!/bin/bash
# Submit a 2-GPU training test run for the Swin speed study, mirroring
# ../run-ace-train.sh but pinned to a chosen cluster so throughput is
# comparable across runs. Usage:
#   ./run-swin-train-test.sh <config filename in this dir> <job_name containing "test"> <cluster>

set -e
command -v gantry >/dev/null || export PATH="$HOME/mamba/envs/fme/bin:$PATH"

CONFIG_FILENAME="$1"
JOB_NAME="$2"
CLUSTER="$3"
if [[ "$JOB_NAME" != *test* ]]; then
  echo "job name must contain 'test'" >&2
  exit 1
fi
SCRIPT_DIR=configs/experiments/2026-10-19-fm/swin_speed
REPO_ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
cd "$REPO_ROOT"
CONFIG_PATH="$SCRIPT_DIR/$CONFIG_FILENAME"
N_GPUS=2

python -m fme.ace.validate_config --config_type train "$CONFIG_PATH"

extra_args=()
while IFS= read -r line; do
  [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
done < "$CONFIG_PATH"

gantry run \
  --name "$JOB_NAME" \
  --task-name "$JOB_NAME" \
  --description 'Swin speed training test' \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace \
  --priority "${BEAKER_PRIORITY:-normal}" \
  --cluster "$CLUSTER" \
  --env WANDB_USERNAME=yyexela \
  --env WANDB_NAME="$JOB_NAME" \
  --env WANDB_JOB_TYPE=training \
  --env WANDB_RUN_GROUP=ace2-fm-test \
  --env WANDB_PROJECT=FM \
  --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset-secret google-credentials:/tmp/google_application_credentials.json \
  --gpus $N_GPUS \
  --shared-memory 200GiB \
  --weka climate-default:/climate-default \
  --budget ai2/atec-climate \
  --system-python \
  --install "pip install --no-deps ." \
  --allow-dirty \
  "${extra_args[@]}" \
  -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train "$CONFIG_PATH"
