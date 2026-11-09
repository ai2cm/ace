#!/bin/bash
# Submit one anneal training job (create_anneal_jobs.py config) via gantry.
#
#   run-ace-anneal.sh CONFIG_FILENAME JOB_NAME JOB_GROUP PARENT_DATASET DESCRIPTION
#
# Identical to run-ace-train.sh (image, clusters, GPUs, shared memory, budget,
# weka mount, env and secrets, torchrun entry point) except that it mounts the
# parent run's checkpoint PARENT_CHECKPOINT (default
# training_checkpoints/ema_ckpt_0150.tar) from PARENT_DATASET at
# /parent_ckpt.tar (stepper_training.parameter_init.weights_path) and takes the
# Beaker experiment description as an argument.
#
# Also used by submit_fm_v3_jobs.py, which sets PARENT_CHECKPOINT to the v2
# parent's training_checkpoints/ema_ckpt_0040.tar and N_GPUS=4.

set -e

CONFIG_FILENAME="$1"
JOB_NAME="$2"
JOB_GROUP="$3"
PARENT_DATASET="$4"
DESCRIPTION="$5"
if [[ -z "$DESCRIPTION" ]]; then
  echo "usage: $0 CONFIG_FILENAME JOB_NAME JOB_GROUP PARENT_DATASET DESCRIPTION" >&2
  exit 2
fi
PARENT_CHECKPOINT=${PARENT_CHECKPOINT:-training_checkpoints/ema_ckpt_0150.tar}
MOUNTED_PARENT_CHECKPOINT=/parent_ckpt.tar

SCRIPT_PATH=$(git rev-parse --show-prefix)  # cwd (the experiment root, set by the submit scripts) relative to the repo root
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
 # since we use a service account API key for wandb, we use the beaker username to set the wandb username by default
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
WANDB_PROJECT=${WANDB_PROJECT:-FM}
BEAKER_WORKSPACE=${BEAKER_WORKSPACE:-ai2/ace}
BEAKER_CLUSTER=${BEAKER_CLUSTER:-"ai2/jupiter ai2/titan ai2/saturn"}
BEAKER_PRIORITY=${BEAKER_PRIORITY:-normal}
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=${N_GPUS:-2}  # submit_fm_v3_jobs.py sets 4
CONFIG_PATH="$SCRIPT_PATH/$CONFIG_FILENAME"

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

python -m fme.ace.validate_config --config_type train "$CONFIG_PATH"

cluster_args=()
for cluster in $BEAKER_CLUSTER; do
  cluster_args+=(--cluster "$cluster")
done

gantry run \
  --name "$JOB_NAME" \
  --task-name "$JOB_NAME" \
  --description "$DESCRIPTION" \
  --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
  --workspace "$BEAKER_WORKSPACE" \
  --priority "$BEAKER_PRIORITY" \
  --min-runtime 4h \
  "${cluster_args[@]}" \
  --env WANDB_USERNAME="$WANDB_USERNAME" \
  --env WANDB_NAME="$JOB_NAME" \
  --env WANDB_JOB_TYPE=training \
  --env WANDB_RUN_GROUP="$JOB_GROUP" \
  --env WANDB_PROJECT="$WANDB_PROJECT" \
  --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset-secret google-credentials:/tmp/google_application_credentials.json \
  --dataset "$PARENT_DATASET:$PARENT_CHECKPOINT:$MOUNTED_PARENT_CHECKPOINT" \
  --gpus $N_GPUS \
  --shared-memory 200GiB \
  --weka climate-default:/climate-default \
  --budget ai2/atec-climate \
  --system-python \
  --install "pip install --no-deps ." \
  --allow-dirty \
  -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_PATH
