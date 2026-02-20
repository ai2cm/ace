#!/opt/homebrew/bin/bash

set -e

WANDB_USERNAME=spencerc_ai2
CONFIG_FILENAME="ace-amip-inference-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME

ENSEMBLE_ID="ic_0002"
AMIP_DATA_ROOT="/climate-default/2026-01-28-vertically-resolved-c96-1deg-shield-amip-ensemble-dataset"

declare -A INITIAL_CONDITIONS
INITIAL_CONDITIONS=( \
    ["1"]="1979-01-01T06:00:00" \
    ["2"]="1979-01-01T12:00:00" \
    ["3"]="1979-01-01T18:00:00" \
    ["4"]="1979-01-02T00:00:00" \
    ["5"]="1979-01-02T06:00:00" \
)

declare -A MODELS=( \
    [published-baseline-rs3]="01J4BR6J5AW32ZDQ77VZ60P4KT" \
)

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT  # so config path is valid no matter where we are running this script

SPIN_UP_MAXIMUM_N_FORWARD_STEPS=1459
SPIN_UP_EXPERIMENT_DIR="/results/spin-up"

MAIN_INITIAL_CONDITION_TIME="1980-01-01T00:00:00"
MAIN_INITIAL_CONDITION_PATH="/results/spin-up/restart.nc"
MAIN_N_FORWARD_STEPS=59904
MAIN_EXPERIMENT_DIR="/results/main"

for model in "${!MODELS[@]}"; do
    dataset_id="${MODELS[$model]}"

    for initial_condition in "${!INITIAL_CONDITIONS[@]}"; do
        spin_up_initial_condition_time="${INITIAL_CONDITIONS[$initial_condition]}"
        spin_up_n_forward_steps="$((SPIN_UP_MAXIMUM_N_FORWARD_STEPS - initial_condition + 1))"
        spin_up_log_to_wandb=false  # Disable logging to wandb in spin up case.

        job_name=2026-02-20-$model-amip-ic$initial_condition
        spin_up_overrides="\
            experiment_dir=$SPIN_UP_EXPERIMENT_DIR \
            forcing_loader.dataset.data_path=$AMIP_DATA_ROOT \
            initial_condition.path=$AMIP_DATA_ROOT/${ENSEMBLE_ID}.zarr \
            initial_condition.engine=zarr \
            initial_condition.start_indices.times=[$spin_up_initial_condition_time] \
            n_forward_steps=$spin_up_n_forward_steps \
            logging.log_to_wandb=$spin_up_log_to_wandb \
        "
        main_overrides="\
            experiment_dir=$MAIN_EXPERIMENT_DIR \
            forcing_loader.dataset.data_path=$AMIP_DATA_ROOT \
            forcing_loader.dataset.engine=zarr \
            forcing_loader.dataset.file_pattern=${ENSEMBLE_ID}.zarr \
            initial_condition.path=$MAIN_INITIAL_CONDITION_PATH \
            initial_condition.start_indices.times=[$MAIN_INITIAL_CONDITION_TIME] \
            n_forward_steps=$MAIN_N_FORWARD_STEPS \
        "

        python -m fme.ace.validate_config --config_type inference $CONFIG_PATH --override $spin_up_overrides
        python -m fme.ace.validate_config --config_type inference $CONFIG_PATH --override $main_overrides

        gantry run \
            --name $job_name \
            --description 'Run inference with ACE' \
            --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
            --workspace ai2/ace \
            --priority high \
            --not-preemptible \
            --cluster ai2/jupiter \
            --cluster ai2/titan \
            --cluster ai2/ceres \
            --env WANDB_USERNAME=$WANDB_USERNAME \
            --env WANDB_NAME=$job_name \
            --env WANDB_JOB_TYPE=inference \
            --env WANDB_RUN_GROUP= \
            --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
            --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
            --dataset-secret google-credentials:/tmp/google_application_credentials.json \
            --dataset $dataset_id:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar \
            --gpus 1 \
            --shared-memory 20GiB \
            --weka climate-default:/climate-default \
            --budget ai2/climate \
            --system-python \
            --install "pip install --no-deps ." \
            -- /bin/bash -c "\
                python -I -m fme.ace.inference $CONFIG_PATH --override $spin_up_overrides \
                && \
                python -I -m fme.ace.inference $CONFIG_PATH --override $main_overrides \
            "
    done
done
