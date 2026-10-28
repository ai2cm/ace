#!/bin/bash
# Submit one inference job on a training checkpoint via gantry.
#
#   run-ace-inference-job.sh MODE CONFIG_FILENAME JOB_NAME JOB_GROUP \
#       RESULTS_DATASET [CHECKPOINT_PATH]
#
# MODE selects the entry point and how the config is validated first:
#   suite      run_eval_suite.py (evaluator suite, one wandb run)
#   evaluator  python -m fme.ace.evaluator
#   inference  python -m fme.ace.inference
# CHECKPOINT_PATH inside RESULTS_DATASET is mounted at /ckpt.tar.
# Set SKIP_VALIDATE=1 when the caller has validated the config already.
# Ported from the 2026-06-26 run-ace-eval.sh / run-ace-evaluator.sh /
# run-ace-inference.sh. The gantry options follow run-ace-train.sh (same
# defaults, --min-runtime, budget, weka, env and secrets, and `# arg:` extra
# gantry args read from the config header); they differ only where inference
# needs something else: 1 GPU and 50GiB shared memory instead of 2 GPUs and
# 200GiB, WANDB_JOB_TYPE=inference, the checkpoint dataset mount, and a
# single-process python entry point instead of torchrun.

set -e

MODE="$1"
CONFIG_FILENAME="$2"
JOB_NAME="$3"
JOB_GROUP="$4"
EXISTING_RESULTS_DATASET="$5"
CHECKPOINT_PATH="${6:-training_checkpoints/best_inference_ckpt.tar}"
if [[ -z "$EXISTING_RESULTS_DATASET" ]]; then
  echo "usage: $0 MODE CONFIG_FILENAME JOB_NAME JOB_GROUP RESULTS_DATASET [CHECKPOINT_PATH]" >&2
  exit 2
fi
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
# since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
WANDB_PROJECT=${WANDB_PROJECT:-FM}
BEAKER_WORKSPACE=${BEAKER_WORKSPACE:-ai2/ace}
BEAKER_CLUSTER=${BEAKER_CLUSTER:-"ai2/jupiter ai2/titan ai2/saturn"}
BEAKER_PRIORITY=${BEAKER_PRIORITY:-normal}
REPO_ROOT=$(git rev-parse --show-toplevel)

case "$MODE" in
  suite)
    VALIDATE=(python "$SCRIPT_PATH/run_eval_suite.py" --validate-only "$CONFIG_PATH")
    COMMAND=(python -I "$SCRIPT_PATH/run_eval_suite.py" "$CONFIG_PATH")
    ;;
  evaluator)
    VALIDATE=(python -m fme.ace.validate_config --config_type evaluator "$CONFIG_PATH")
    COMMAND=(python -I -m fme.ace.evaluator "$CONFIG_PATH")
    ;;
  inference)
    VALIDATE=(python -m fme.ace.validate_config --config_type inference "$CONFIG_PATH")
    COMMAND=(python -I -m fme.ace.inference "$CONFIG_PATH")
    ;;
  *)
    echo "unknown MODE '$MODE' (suite|evaluator|inference)" >&2
    exit 2
    ;;
esac

cd "$REPO_ROOT"  # so config path is valid no matter where we are running this script

if [[ "${SKIP_VALIDATE:-0}" != "1" ]]; then
  "${VALIDATE[@]}"
fi

# Extract additional gantry args from the config header, as run-ace-train.sh.
extra_args=()
while IFS= read -r line; do
  [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
done < "$CONFIG_PATH"

cluster_args=()
for cluster in $BEAKER_CLUSTER; do
  cluster_args+=(--cluster "$cluster")
done

gantry run \
  --name "$JOB_NAME" \
  --task-name "$JOB_NAME" \
  --description "Run ACE FM $MODE on a 2026-10-19 checkpoint" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace "$BEAKER_WORKSPACE" \
  --priority "$BEAKER_PRIORITY" \
  --min-runtime 4h \
  "${cluster_args[@]}" \
  --env WANDB_USERNAME="$WANDB_USERNAME" \
  --env WANDB_NAME="$JOB_NAME" \
  --env WANDB_JOB_TYPE=inference \
  --env WANDB_RUN_GROUP="$JOB_GROUP" \
  --env WANDB_PROJECT="$WANDB_PROJECT" \
  --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset-secret google-credentials:/tmp/google_application_credentials.json \
  --dataset "$EXISTING_RESULTS_DATASET:$CHECKPOINT_PATH:/ckpt.tar" \
  --gpus 1 \
  --shared-memory 50GiB \
  --weka climate-default:/climate-default \
  --budget ai2/atec-climate \
  --system-python \
  --install "pip install --no-deps ." \
  --allow-dirty \
  "${extra_args[@]}" \
  -- "${COMMAND[@]}"
