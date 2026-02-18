#!/opt/homebrew/bin/bash

set -e

WANDB_USERNAME=spencerc_ai2
CONFIG_FILENAME="ace-som-inference-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME

SPIN_UP_FORCING_ROOT=/climate-default/2024-08-15-vertically-resolved-1deg-c96-shield-som-ensemble-spin-up-fme-dataset/netcdfs
MAIN_FORCING_ROOT=/climate-default/2026-01-28-vertically-resolved-1deg-c96-shield-som-ensemble-fme-dataset

declare -A SPIN_UP_FORCING_DATASETS
SPIN_UP_FORCING_DATASETS=( \
    ["1xCO2"]="$SPIN_UP_FORCING_ROOT/concatenated-1xCO2-ic_0005" \
    # ["2xCO2"]="$SPIN_UP_FORCING_ROOT/concatenated-2xCO2-ic_0005" \
    # ["3xCO2"]="$SPIN_UP_FORCING_ROOT/concatenated-3xCO2-ic_0002" \
    # ["4xCO2"]="$SPIN_UP_FORCING_ROOT/concatenated-4xCO2-ic_0005" \
)

declare -A MAIN_FORCING_DATASETS
MAIN_FORCING_DATASETS=( \
    ["1xCO2"]="1xCO2-ic_0005.zarr" \
    # ["2xCO2"]="2xCO2-ic_0005.zarr" \
    # ["3xCO2"]="3xCO2-ic_0002.zarr" \
    # ["4xCO2"]="4xCO2-ic_0005.zarr" \
)

# For stochastic models we do not necessarily need to use this staggered
# initialization approach, but we retain it for compatibility with the
# initialization approach needed for deterministic models.
declare -A INITIAL_CONDITIONS
INITIAL_CONDITIONS=( \
    ["1"]="2030-01-01T06:00:00" \
    # ["2"]="2030-01-01T12:00:00" \
    # ["3"]="2030-01-01T18:00:00" \
    # ["4"]="2030-01-02T00:00:00" \
    # ["5"]="2030-01-02T06:00:00" \
)

declare -A MODELS=( \
    [published-baseline-rs3]="01J4BR6J5AW32ZDQ77VZ60P4KT" \
    # [no-random-co2-rs0]="01KHGDAMB2BDZQS8JFF65A2YDR" \
    # [no-random-co2-rs1]="01KH4SDCYN1NF2RP2JXZS0WZ1Y" \
    # [no-random-co2-energy-conserving-rs0]="01KHGDA8TVGP9JKWVJ1N0SMHCN" \
    # [no-random-co2-energy-conserving-rs1]="01KH4SDT1Q5246GZ307W8AW4M3" \
)

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT  # so config path is valid no matter where we are running this script

SPIN_UP_MAXIMUM_N_FORWARD_STEPS=1460
SPIN_UP_EXPERIMENT_DIR="/results/spin-up"

MAIN_INITIAL_CONDITION_TIME="2031-01-01T06:00:00"
MAIN_INITIAL_CONDITION_PATH="/results/spin-up/restart.nc"
MAIN_N_FORWARD_STEPS=14604
MAIN_EXPERIMENT_DIR="/results/main"

for model in "${!MODELS[@]}"; do
    dataset_id="${MODELS[$model]}"

    for climate in "${!SPIN_UP_FORCING_DATASETS[@]}"; do
        spin_up_forcing_path="${SPIN_UP_FORCING_DATASETS[$climate]}"
        spin_up_initial_condition_path=$spin_up_forcing_path/2030010100.nc
        main_forcing_path="${MAIN_FORCING_DATASETS[$climate]}"

        for initial_condition in "${!INITIAL_CONDITIONS[@]}"; do
            spin_up_initial_condition_time="${INITIAL_CONDITIONS[$initial_condition]}"
            spin_up_n_forward_steps="$((SPIN_UP_MAXIMUM_N_FORWARD_STEPS - initial_condition - 1))"
            spin_up_log_to_wandb=false  # Disable logging to wandb in spin up case.

            job_name=2026-02-18-$model-$climate-ic$initial_condition
            spin_up_overrides="\
                experiment_dir=$SPIN_UP_EXPERIMENT_DIR \
                forcing_loader.dataset.data_path=$spin_up_forcing_path \
                initial_condition.path=$spin_up_initial_condition_path \
                initial_condition.start_indices.times=[$spin_up_initial_condition_time] \
                n_forward_steps=$spin_up_n_forward_steps \
                logging.log_to_wandb=$spin_up_log_to_wandb \
            "
            main_overrides="\
                experiment_dir=$MAIN_EXPERIMENT_DIR \
                forcing_loader.dataset.data_path=$MAIN_FORCING_ROOT \
                forcing_loader.dataset.engine=zarr \
                forcing_loader.dataset.file_pattern=$main_forcing_path \
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
                --cluster ai2/jupiter \
                --env WANDB_USERNAME=$WANDB_USERNAME \
                --env WANDB_NAME=$job_name \
                --env WANDB_JOB_TYPE=inference \
                --env WANDB_RUN_GROUP= \
                --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
                --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
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
done
