#!/bin/bash

set -e

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <config_subdirectory> [--atmos_stats <path>] [--ocean_stats <path>]"
  echo "  - <config_subdirectory>: Subdirectory containing the 'train-config.yaml' to use."
  echo "  - --atmos_stats: Override atmosphere stats data path (optional)"
  echo "  - --ocean_stats: Override ocean stats data path (optional)"
  exit 1
fi

# The subdirectory (passed as an argument) that holds the config file.
CONFIG_SUBDIR=$1
shift

# Parse optional arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --atmos_stats)
      ATMOS_STATS_DATA="$2"
      shift 2
      ;;
    --ocean_stats)
      OCEAN_STATS_DATA="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Get the absolute directory where this script is located.
SCRIPT_DIR=$(git rev-parse --show-prefix)

# Construct the full path to the specified configuration file.
CONFIG_FILENAME="train-config.yaml"
CONFIG_PATH="$SCRIPT_DIR/$CONFIG_SUBDIR/$CONFIG_FILENAME"

# Since we use a service account API key for wandb, we use the beaker username to set the wandb username.
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Set default values if not provided via CLI args
if [[ -z "$ATMOS_STATS_DATA" ]]; then
    ATMOS_STATS_DATA=jamesd/2025-08-22-cm4-piControl-200yr-coupled-stats-atmosphere
fi
if [[ -z "$OCEAN_STATS_DATA" ]]; then
    OCEAN_STATS_DATA=jamesd/2025-08-22-cm4-piControl-200yr-coupled-stats-ocean
fi

echo
echo "Using the following stats:"
echo " - Atmosphere: ${ATMOS_STATS_DATA}"
echo " - Ocean: ${OCEAN_STATS_DATA}"

# Change to the repo root so paths are valid no matter where we run the script from.
cd "$REPO_ROOT"

while read PRETRAINING; do
    GROUP=$(echo "$PRETRAINING" | cut -d"|" -f1)
    OCEAN_PROJECT=$(echo "$PRETRAINING" | cut -d"|" -f2)
    OCEAN_WANDB_ID=$(echo "$PRETRAINING" | cut -d"|" -f3)
    OCEAN_CKPT=$(echo "$PRETRAINING" | cut -d"|" -f4)
    ATMOS_PROJECT=$(echo "$PRETRAINING" | cut -d"|" -f5)
    ATMOS_WANDB_ID=$(echo "$PRETRAINING" | cut -d"|" -f6)
    ATMOS_CKPT=$(echo "$PRETRAINING" | cut -d"|" -f7)
    STATUS=$(echo "$PRETRAINING" | cut -d"|" -f8)
    PRIORITY=$(echo "$PRETRAINING" | cut -d"|" -f9)
    CLUSTER=$(echo "$PRETRAINING" | cut -d"|" -f10)
    N_GPUS=$(echo "$PRETRAINING" | cut -d"|" -f11)
    SHARED_MEM=$(echo "$PRETRAINING" | cut -d"|" -f12)
    RETRIES=$(echo "$PRETRAINING" | cut -d"|" -f13)
    WORKSPACE=$(echo "$PRETRAINING" | cut -d"|" -f14)
    OVERRIDE_ARGS=$(echo "$PRETRAINING" | cut -d"|" -f15)
    if [[ "$STATUS" != "train" ]]; then
        continue
    fi
    if [[ -z $RETRIES ]]; then
        RETRIES=0
    fi
    JOB_GROUP="${GROUP}"
    JOB_NAME="${JOB_GROUP}-train"
    ATMOS_EXPER_ID=$(
        python $REPO_ROOT/scripts/wandb/wandb_to_beaker_experiment.py \
            --project "$ATMOS_PROJECT" --wandb_id "$ATMOS_WANDB_ID"
    )
    OCEAN_EXPER_ID=$(
        python $REPO_ROOT/scripts/wandb/wandb_to_beaker_experiment.py \
            --project "$OCEAN_PROJECT" --wandb_id "$OCEAN_WANDB_ID"
    )
    EXISTING_RESULTS_ATMOS_DATASET=$(
        beaker experiment get $ATMOS_EXPER_ID --format json |
            jq '.[].jobs[-1].result' | grep "beaker" | cut -d'"' -f4
    )
    EXISTING_RESULTS_OCEAN_DATASET=$(
        beaker experiment get $OCEAN_EXPER_ID --format json |
            jq '.[].jobs[-1].result' | grep "beaker" | cut -d'"' -f4
    )
    declare -a CLUSTER_ARGS
    if [[ "$CLUSTER" == "titan" ]]; then
        if [[ -z "$WORKSPACE" ]]; then
            WORKSPACE=ai2/climate-titan
        fi
        CLUSTER_ARGS=(
            --workspace "$WORKSPACE"
            --cluster ai2/titan-cirrascale
        )
    else
        if [[ -z "$WORKSPACE" ]]; then
            WORKSPACE=ai2/climate-ceres
        fi
        CLUSTER_ARGS=(
            --workspace "$WORKSPACE"
            --cluster ai2/ceres-cirrascale
        )
    fi

    TEMPLATE_CONFIG_FILENAME="train-config-template.yaml"

    # get the template from the config subdir
    TEMPLATE_CONFIG_PATH="$SCRIPT_DIR/$CONFIG_SUBDIR/$TEMPLATE_CONFIG_FILENAME"

    bash $REPO_ROOT/scripts/coupled/create_coupled_train_config.sh \
        "$EXISTING_RESULTS_ATMOS_DATASET" \
        "$EXISTING_RESULTS_OCEAN_DATASET" \
        "$TEMPLATE_CONFIG_PATH" \
        "$CONFIG_PATH"

    if [[ -n "${OVERRIDE_ARGS}" ]]; then
        OVERRIDE="--override ${OVERRIDE_ARGS}"
    else
        OVERRIDE=""
    fi

    echo
    echo "Launching coupled training job:"
    echo " - Job name: ${JOB_NAME}"
    echo " - Config: ${CONFIG_PATH}"
    echo " - Atmosphere training experiment ID: ${ATMOS_EXPER_ID}"
    echo " - Atmosphere results dataset ID: ${EXISTING_RESULTS_ATMOS_DATASET}"
    echo " - Atmosphere checkpoint type: ${ATMOS_CKPT}"
    echo " - Ocean training experiment ID: ${OCEAN_EXPER_ID}"
    echo " - Ocean results dataset ID: ${EXISTING_RESULTS_OCEAN_DATASET}"
    echo " - Ocean checkpoint type: ${OCEAN_CKPT}"
    echo " - Priority: ${PRIORITY}"
    echo " - Cluster: ${CLUSTER} (${RETRIES} retries)"
    echo " - GPUs: ${N_GPUS}"
    echo " - Shared memory: ${SHARED_MEM}"
    echo " - Override: ${OVERRIDE_ARGS}"

    python -m fme.coupled.validate_config "$CONFIG_PATH" --config_type train $OVERRIDE

    if git status --porcelain "$CONFIG_PATH" | grep -q .; then
        git add "$CONFIG_PATH"
        git commit -m"${JOB_NAME}"
        git push origin "${GIT_BRANCH}"
    fi
    echo

    EXPERIMENT_ID=$(
        gantry run \
            --name $JOB_NAME \
            --description "Run coupled training from uncoupled pretraining: ${JOB_GROUP}" \
            --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
            --priority $PRIORITY \
            --preemptible \
            --retries $RETRIES \
            "${CLUSTER_ARGS[@]}" \
            --weka climate-default:/climate-default \
            --env WANDB_USERNAME=$BEAKER_USERNAME \
            --env WANDB_NAME=$JOB_NAME \
            --env WANDB_JOB_TYPE=training \
            --env WANDB_RUN_GROUP=$JOB_GROUP \
            --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
            --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
            --dataset-secret google-credentials:/tmp/google_application_credentials.json \
            --dataset $ATMOS_STATS_DATA:/atmos_stats \
            --dataset $OCEAN_STATS_DATA:/ocean_stats \
            --dataset $EXISTING_RESULTS_ATMOS_DATASET:training_checkpoints/"$ATMOS_CKPT".tar:/atmos_ckpt.tar \
            --dataset $EXISTING_RESULTS_OCEAN_DATASET:training_checkpoints/"$OCEAN_CKPT".tar:/ocean_ckpt.tar \
            --gpus $N_GPUS \
            --shared-memory "${SHARED_MEM}" \
            --budget ai2/climate \
            --no-conda \
            --install "pip install --no-deps ." \
            -- torchrun --nproc_per_node $N_GPUS -m fme.coupled.train "$CONFIG_PATH" $OVERRIDE |
            tee /dev/tty |
            grep beaker.org |
            cut -d/ -f5
    )

    # remove or change 'training' once completed in order to submit an evaluator job
    { echo;
      echo "${JOB_GROUP}|${EXPERIMENT_ID}|training|best_inference_ckpt|normal|--not-preemptible";
    } >> "${SCRIPT_DIR}/${CONFIG_SUBDIR}/experiments.txt"

    git add "${SCRIPT_DIR}/${CONFIG_SUBDIR}/experiments.txt"
    git commit -m"Update coupled/${CONFIG_SUBDIR}/experiments.txt"
    git push origin "${GIT_BRANCH}"

done <"${SCRIPT_DIR}/${CONFIG_SUBDIR}/pretraining.txt"
