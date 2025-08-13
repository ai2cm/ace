#!/bin/bash

set -e

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <config_subdirectory>"
  echo "  - <config_subdirectory>: Subdirectory containing the 'finetune-config.yaml' to use."
  exit 1
fi

# The subdirectory (passed as an argument) that holds the config file.
CONFIG_SUBDIR=$1

# Get the absolute directory where this script is located.
SCRIPT_DIR=$(git rev-parse --show-prefix)

# Construct the full path to the specified configuration file.
CONFIG_FILENAME="finetune-config.yaml"
CONFIG_PATH="$SCRIPT_DIR/$CONFIG_SUBDIR/$CONFIG_FILENAME"

# Since we use a service account API key for wandb, we use the beaker username to set the wandb username.
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# FIXME: this needs to be per-task configurable
ATMOS_STATS_DATA=jamesd/2025-08-07-cm4-piControl-200yr-coupled-stats-atmosphere
OCEAN_STATS_DATA=jamesd/2025-08-07-cm4-piControl-200yr-coupled-stats-ocean

if [[ "$CONFIG_SUBDIR" == "ft" ]] || [[ "$CONFIG_SUBDIR" == "fto" ]]; then
    echo "FIXME: incorrect statsdata, exiting"
    exit 1
fi

# Change to the repo root so paths are valid no matter where we run the script from.
cd "$REPO_ROOT"


TEMPLATE_CONFIG_FILENAME="finetune-config-template.yaml"

# get the template from the config subdir
TEMPLATE_CONFIG_PATH="$SCRIPT_DIR/$CONFIG_SUBDIR/$TEMPLATE_CONFIG_FILENAME"

while read PRETRAINING; do
    GROUP=$(echo "$PRETRAINING" | cut -d"|" -f1)
    WANDB_PROJECT=$(echo "$PRETRAINING" | cut -d"|" -f2)
    WANDB_ID=$(echo "$PRETRAINING" | cut -d"|" -f3)
    CKPT_TYPE=$(echo "$PRETRAINING" | cut -d"|" -f4)
    STATUS=$(echo "$PRETRAINING" | cut -d"|" -f5)
    PRIORITY=$(echo "$PRETRAINING" | cut -d"|" -f6)
    CLUSTER=$(echo "$PRETRAINING" | cut -d"|" -f7)
    RETRIES=$(echo "$PRETRAINING" | cut -d"|" -f8)
    OVERRIDE_ARGS=$(echo "$PRETRAINING" | cut -d"|" -f9)
    if [[ "$STATUS" != "train" ]]; then
        continue
    fi
    if [[ -z $RETRIES ]]; then
        RETRIES=0
    fi
    JOB_GROUP="${GROUP}"
    JOB_NAME="${JOB_GROUP}-train"
    EXPER_ID=$(
        python $REPO_ROOT/scripts/wandb/wandb_to_beaker_experiment.py \
            --project "$WANDB_PROJECT" --wandb_id "$WANDB_ID"
    )
    EXISTING_RESULTS_DATASET=$(
        beaker experiment get $EXPER_ID --format json |
            jq '.[].jobs[-1].result' | grep "beaker" | cut -d'"' -f4
    )
    declare -a CLUSTER_ARGS
    if [[ "$CLUSTER" == "titan" ]]; then
        CLUSTER_ARGS=(
            --workspace ai2/climate-titan
            --cluster ai2/titan-cirrascale
        )
        N_GPUS=4
    else
        CLUSTER_ARGS=(
            --workspace ai2/climate-ceres
            --cluster ai2/ceres-cirrascale
        )
        N_GPUS=8
    fi

    bash $SCRIPT_DIR/create_finetune_config.sh \
        "$EXISTING_RESULTS_DATASET" \
        "$TEMPLATE_CONFIG_PATH" \
        "$CONFIG_PATH"

    if [[ -n "${OVERRIDE_ARGS}" ]]; then
        OVERRIDE="--override ${OVERRIDE_ARGS}"
    else
        OVERRIDE=""
    fi

    echo
    echo "Launching coupled fine-tuning job:"
    echo " - Job name: ${JOB_NAME}"
    echo " - Config: ${CONFIG_PATH}"
    echo " - Coupled pretraining experiment ID: ${EXPER_ID}"
    echo " - Checkpoint type: ${CKPT_TYPE}"
    echo " - Priority: ${PRIORITY}"
    echo " - Cluster: ${CLUSTER} (${RETRIES} retries)"
    echo " - Override: ${OVERRIDE_ARGS}"

    python -m fme.ace.validate_config "$CONFIG_PATH" --config_type train $OVERRIDE

    if git status --porcelain "$CONFIG_PATH" | grep -q .; then
        git add "$CONFIG_PATH"
        git commit -m"${JOB_NAME}"
        git push origin "${GIT_BRANCH}"
    fi

    EXPERIMENT_ID=$(
        gantry run \
            --name $JOB_NAME \
            --description "Run ACE coupled fine-tuning" \
            --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
            --priority $PRIORITY \
            --preemptible \
            --retries $RETRIES \
            "${CLUSTER_ARGS[@]}" \
            --weka climate-default:/climate-default \
            --env NCCL_DEBUG=INFO \
            --env WANDB_USERNAME=$BEAKER_USERNAME \
            --env WANDB_NAME=$JOB_NAME \
            --env WANDB_JOB_TYPE=training \
            --env WANDB_RUN_GROUP=$JOB_GROUP \
            --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
            --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
            --dataset-secret google-credentials:/tmp/google_application_credentials.json \
            --dataset $ATMOS_STATS_DATA:/atmos_stats \
            --dataset $OCEAN_STATS_DATA:/ocean_stats \
            --dataset $EXISTING_RESULTS_DATASET:training_checkpoints/"$CKPT_TYPE".tar:/ckpt.tar \
            --gpus $N_GPUS \
            --shared-memory 800GiB \
            --budget ai2/climate \
            --no-conda \
            --install "pip install --no-deps ." \
            -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train "$CONFIG_PATH" $OVERRIDE |
            tee /dev/tty |
            grep beaker.org |
            cut -d/ -f5
    )

    # remove or change 'training' once completed in order to submit an evaluator job
    { echo;
      echo "${JOB_GROUP}|${JOB_TAG}|${EXPERIMENT_ID}|training|best_inference_ckpt|normal|--not-preemptible";
    } >> "${SCRIPT_DIR}/${CONFIG_SUBDIR}/experiments.txt"

    git add "${SCRIPT_DIR}/${CONFIG_SUBDIR}/experiments.txt"
    git commit -m"Update uncoupled/${CONFIG_SUBDIR}/experiments.txt"
    git push origin "${GIT_BRANCH}"

done <"${SCRIPT_DIR}/${CONFIG_SUBDIR}/finetuning.txt"
