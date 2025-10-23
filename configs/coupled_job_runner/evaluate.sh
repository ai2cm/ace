#!/bin/bash
# Wrapper script for evaluation jobs
# Usage: evaluate.sh <experiment_dir> <config_subdirectory>

set -e

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <experiment_dir> <config_subdirectory>"
  echo "  - <experiment_dir>: Path to experiment directory (e.g., experiments/2025-08-08-jamesd/coupled or experiments/2025-08-08-jamesd/uncoupled)"
  echo "  - <config_subdirectory>: Subdirectory containing the evaluator config files"
  exit 1
fi

# Parse positional arguments
EXPERIMENT_DIR="$1"
CONFIG_SUBDIR="$2"

# Set up paths
REPO_ROOT=$(git rev-parse --show-toplevel)
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')

# Determine which fme module to use based on EXPERIMENT_DIR
if [[ "$EXPERIMENT_DIR" =~ coupled ]]; then
    FME_MODULE_VALIDATE="fme.coupled.validate_config"
    FME_MODULE_EVALUATOR="fme.coupled.evaluator"
else
    FME_MODULE_VALIDATE="fme.ace.validate_config"
    FME_MODULE_EVALUATOR="fme.ace.evaluator"
fi

# Construct full paths
FULL_EXPERIMENT_DIR="$REPO_ROOT/$EXPERIMENT_DIR"
INPUT_PATH="$FULL_EXPERIMENT_DIR/$CONFIG_SUBDIR/experiments.txt"

# Change to repo root so paths are valid
cd "$REPO_ROOT"

while read TRAIN_EXPER; do
    JOB_GROUP=$(echo "$TRAIN_EXPER" | cut -d"|" -f1)
    EXPER_ID=$(echo "$TRAIN_EXPER" | cut -d"|" -f2)
    STATUS=$(echo "$TRAIN_EXPER" | cut -d"|" -f3)
    CKPT=$(echo "$TRAIN_EXPER" | cut -d"|" -f4)
    PRIORITY=$(echo "$TRAIN_EXPER" | cut -d"|" -f5)
    PREEMPTIBLE=$(echo "$TRAIN_EXPER" | cut -d"|" -f6)
    OVERRIDE_ARGS=$(echo "$TRAIN_EXPER" | cut -d"|" -f7)
    EXISTING_RESULTS_DATASET=$(echo "$TRAIN_EXPER" | cut -d"|" -f8)
    WORKSPACE=$(echo "$TRAIN_EXPER" | cut -d"|" -f9)

    # Check if STATUS starts with "run_"
    if [[ ! "$STATUS" =~ ^run_ ]]; then
        continue
    fi

    # Derive config tag and filename from STATUS
    # Example: if STATUS is "run_ICx1"
    # CURRENT_CONFIG_TAG becomes "ICx1"
    # CURRENT_CONFIG_FILENAME becomes "evaluator-config-ICx1.yaml"
    CURRENT_CONFIG_TAG=${STATUS#run_}
    CURRENT_CONFIG_FILENAME="evaluator-config-${CURRENT_CONFIG_TAG}.yaml"

    JOB_NAME="${JOB_GROUP}-evaluator_${CKPT}-${CURRENT_CONFIG_TAG}"

    # Construct absolute path for file operations
    CONFIG_PATH="${FULL_EXPERIMENT_DIR}/${CONFIG_SUBDIR}/${CURRENT_CONFIG_FILENAME}"

    # Construct relative path for gantry/python commands
    CONFIG_PATH_REL="${EXPERIMENT_DIR}/${CONFIG_SUBDIR}/${CURRENT_CONFIG_FILENAME}"

    if [[ ! -f "$CONFIG_PATH" ]]; then
        echo "Error: Config file not found at ${CONFIG_PATH} for JOB_NAME: ${JOB_NAME}. Skipping."
        continue
    fi

    if [[ -z $PRIORITY ]]; then
        PRIORITY=normal
    fi

    if [[ -z $PREEMPTIBLE ]]; then
        PREEMPTIBLE=--not-preemptible
    fi

    if [[ -z $EXISTING_RESULTS_DATASET ]]; then
        EXISTING_RESULTS_DATASET=$(beaker experiment get "$EXPER_ID" --format json | jq '.[].jobs[-1].result' | grep "beaker" | cut -d'"' -f4)
    fi

    if [[ -z "$WORKSPACE" ]]; then
        WORKSPACE=ai2/ace
    fi

    echo
    echo "Launching ${EXPERIMENT_DIR} evaluator job:"
    echo " - Config path: ${CONFIG_PATH_REL}"
    echo " - Group: ${JOB_GROUP}"
    echo " - Checkpoint: ${CKPT}"
    echo " - Training results dataset ID: ${EXISTING_RESULTS_DATASET}"
    echo " - Priority: ${PRIORITY}"
    echo " - ${PREEMPTIBLE}"
    echo " - --override args: ${OVERRIDE_ARGS}"

    echo

    # Validate config (use relative path)
    python -m $FME_MODULE_VALIDATE --config_type evaluator "$CONFIG_PATH_REL" --override $OVERRIDE_ARGS

    echo "$JOB_NAME"
    gantry run \
        --name "$JOB_NAME" \
        --description "Run ${EXPERIMENT_DIR} evaluator" \
        --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
        --priority "$PRIORITY" \
        $PREEMPTIBLE \
        --cluster ceres \
        --workspace "$WORKSPACE" \
        --weka climate-default:/climate-default \
        --env WANDB_USERNAME="$BEAKER_USERNAME" \
        --env WANDB_NAME="$JOB_NAME" \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP="$JOB_GROUP" \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset "${EXISTING_RESULTS_DATASET}:training_checkpoints/${CKPT}.tar:/ckpt.tar" \
        --gpus 1 \
        --shared-memory 20GiB \
        --budget ai2/climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- python -I -m $FME_MODULE_EVALUATOR "$CONFIG_PATH_REL" --override $OVERRIDE_ARGS
    echo
done <"$INPUT_PATH"
