#!/bin/bash
# Wrapper script for evaluation jobs
# Usage: evaluate.sh <experiment_dir> <config_subdirectory> [--dry-run]

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the shared library
source "$SCRIPT_DIR/lib.sh"

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 <experiment_dir> <config_subdirectory> [--dry-run]"
  echo "  - <experiment_dir>: Path to experiment directory (e.g., experiments/2025-08-08-jamesd/coupled or experiments/2025-08-08-jamesd/uncoupled)"
  echo "  - <config_subdirectory>: Subdirectory containing the evaluator config files"
  echo "  - --dry-run: Preview actions without launching jobs"
  exit 1
fi

# Parse positional arguments
EXPERIMENT_DIR="$1"
CONFIG_SUBDIR="$2"
shift 2

# Parse dry-run flag
parse_dry_run_flag "$@"

# Initialize script environment
init_script_environment

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

# Print dry-run header (no stats for evaluator)
if [[ "$DRY_RUN" == "true" ]]; then
    echo "========================================"
    echo "DRY RUN MODE - No jobs will be launched"
    echo "========================================"
    echo
    echo "Environment:"
    echo "  Repository Root: $REPO_ROOT"
    echo "  Git Branch: $GIT_BRANCH"
    echo "  Beaker Username: $BEAKER_USERNAME"
    echo
fi

# Change to repo root so paths are valid
cd "$REPO_ROOT"

# Initialize counters for dry-run summary
TOTAL_JOBS=0
PROCESSED_JOBS=0
SKIPPED_JOBS=0
FIRST_JOB_PRINTED=false

while read TRAIN_EXPER; do
    TOTAL_JOBS=$((TOTAL_JOBS + 1))
    JOB_GROUP=$(echo "$TRAIN_EXPER" | cut -d"|" -f1)
    TAG=$(echo "$TRAIN_EXPER" | cut -d"|" -f2)
    EXPER_ID=$(echo "$TRAIN_EXPER" | cut -d"|" -f3)
    STATUS=$(echo "$TRAIN_EXPER" | cut -d"|" -f4)
    CKPT=$(echo "$TRAIN_EXPER" | cut -d"|" -f5)
    PRIORITY=$(echo "$TRAIN_EXPER" | cut -d"|" -f6)
    PREEMPTIBLE=$(echo "$TRAIN_EXPER" | cut -d"|" -f7)
    OVERRIDE_ARGS=$(echo "$TRAIN_EXPER" | cut -d"|" -f8)
    EXISTING_RESULTS_DATASET=$(echo "$TRAIN_EXPER" | cut -d"|" -f9)
    WORKSPACE=$(echo "$TRAIN_EXPER" | cut -d"|" -f10)
    CLUSTER=$(echo "$TRAIN_EXPER" | cut -d"|" -f11)

    # Check if STATUS starts with "run_"
    if [[ ! "$STATUS" =~ ^run_ ]]; then
        SKIPPED_JOBS=$((SKIPPED_JOBS + 1))
        continue
    fi

    PROCESSED_JOBS=$((PROCESSED_JOBS + 1))

    # Derive config tag and filename from STATUS
    # Example: if STATUS is "run_ICx1"
    # CURRENT_CONFIG_TAG becomes "ICx1"
    # CURRENT_CONFIG_FILENAME becomes "evaluator-config-ICx1.yaml"
    CURRENT_CONFIG_TAG=${STATUS#run_}
    CURRENT_CONFIG_FILENAME="evaluator-config-${CURRENT_CONFIG_TAG}.yaml"

    JOB_GROUP="${JOB_GROUP}-eval_${CKPT}-${CURRENT_CONFIG_TAG}"

    # Construct JOB_NAME using TAG if present
    if [[ -n "$TAG" ]]; then
        JOB_NAME="${JOB_GROUP}-${TAG}"
    else
        JOB_NAME="${JOB_GROUP}"
    fi

    # Construct absolute path for file operations
    CONFIG_PATH="${FULL_EXPERIMENT_DIR}/${CONFIG_SUBDIR}/${CURRENT_CONFIG_FILENAME}"

    # Construct relative path for gantry/python commands
    CONFIG_PATH_REL="${EXPERIMENT_DIR}/${CONFIG_SUBDIR}/${CURRENT_CONFIG_FILENAME}"

    if [[ ! -f "$CONFIG_PATH" ]]; then
        echo "Error: Config file not found at ${CONFIG_PATH} for JOB_NAME: ${JOB_NAME}. Skipping."
        PROCESSED_JOBS=$((PROCESSED_JOBS - 1))
        SKIPPED_JOBS=$((SKIPPED_JOBS + 1))
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

    if [[ -z "$CLUSTER" ]]; then
        CLUSTER="a100+h100"
    fi

    # Set dummy variables for print functions
    GROUP="$JOB_GROUP"
    N_GPUS=1
    SHARED_MEM="20GiB"
    FME_MODULE="$FME_MODULE_EVALUATOR"

    build_cluster_args "$CLUSTER" "$WORKSPACE"

    # Print job info based on dry-run mode
    if [[ "$DRY_RUN" == "true" ]]; then
        if [[ "$FIRST_JOB_PRINTED" == "false" ]]; then
            print_detailed_job_info
            FIRST_JOB_PRINTED=true
        else
            print_condensed_job_info "$JOB_NAME" "$CONFIG_PATH_REL" "$CLUSTER" "$N_GPUS" "$SHARED_MEM" "$PRIORITY"
        fi
    else
        echo
        echo "Launching ${EXPERIMENT_DIR} evaluator job:"
        echo " - Config path: ${CONFIG_PATH_REL}"
        echo " - Group: ${JOB_GROUP}"
        echo " - Checkpoint: ${CKPT}"
        echo " - Training results dataset ID: ${EXISTING_RESULTS_DATASET}"
        echo " - Cluster: ${CLUSTER}"
        echo " - Priority: ${PRIORITY}"
        echo " - ${PREEMPTIBLE}"
        echo " - --override args: ${OVERRIDE_ARGS}"

        echo
    fi

    # Validate config (use relative path)
    python -m $FME_MODULE_VALIDATE --config_type evaluator "$CONFIG_PATH_REL" --override $OVERRIDE_ARGS

    # Run gantry command unless in dry-run mode
    if [[ "$DRY_RUN" != "true" ]]; then
        echo "$JOB_NAME"
        gantry run \
            --name "$JOB_NAME" \
            --description "Run ${EXPERIMENT_DIR} evaluator" \
            --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
            --priority "$PRIORITY" \
            $PREEMPTIBLE \
            "${CLUSTER_ARGS[@]}" \
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
    fi
done <"$INPUT_PATH"

# Print dry-run summary
if [[ "$DRY_RUN" == "true" ]]; then
    echo
    echo "----------------------------------------"
    echo "SUMMARY"
    echo "----------------------------------------"
    echo "Total Jobs in File: $TOTAL_JOBS"
    echo "  - Will Process: $PROCESSED_JOBS (STATUS=run_*)"
    echo "  - Will Skip: $SKIPPED_JOBS (STATUS!=run_*)"
    echo
    echo "Actions that WOULD be taken:"
    echo "  - Launch $PROCESSED_JOBS evaluation jobs"
    echo
    echo "========================================"
    echo "DRY RUN COMPLETE - No changes were made"
    echo "========================================"
fi
