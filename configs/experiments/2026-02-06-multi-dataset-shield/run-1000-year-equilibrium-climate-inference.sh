#!/opt/homebrew/bin/bash

set -e

DATE="2026-05-21"
WANDB_USERNAME=spencerc_ai2
CONFIG_FILENAME="ace-som-1000-year-inference-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME

INITIAL_CONDITION_ROOT=/climate-default/2026-01-28-vertically-resolved-1deg-c96-shield-som-ensemble-fme-dataset
INITIAL_CONDITION_TIME=2032-01-01T00:00:00

declare -A INITIAL_CONDITION_DATASETS
INITIAL_CONDITION_DATASETS=( \
    # ["1xCO2"]="${INITIAL_CONDITION_ROOT}/1xCO2-ic_0005.zarr" \
    ["2xCO2"]="${INITIAL_CONDITION_ROOT}/2xCO2-ic_0005.zarr" \
    ["3xCO2"]="${INITIAL_CONDITION_ROOT}/3xCO2-ic_0002.zarr" \
    ["4xCO2"]="${INITIAL_CONDITION_ROOT}/4xCO2-ic_0005.zarr" \
)

declare -A CO2_CONCENTRATIONS
CO2_CONCENTRATIONS=( \
    ["1xCO2"]=0.00036343 \
    ["2xCO2"]=0.00072686 \
    ["3xCO2"]=0.00109029 \
    ["4xCO2"]=0.00145372 \
)

declare -A MODELS=( \
    [published-baseline-rs3]="01J4BR6J5AW32ZDQ77VZ60P4KT" \
    # [no-random-co2-rs0]="01KHGDAMB2BDZQS8JFF65A2YDR" \
    # [no-random-co2-rs1]="01KH4SDCYN1NF2RP2JXZS0WZ1Y" \
    # [no-random-co2-energy-conserving-rs0]="01KHGDA8TVGP9JKWVJ1N0SMHCN" \
    # [no-random-co2-energy-conserving-rs1]="01KH4SDT1Q5246GZ307W8AW4M3" \
    # [full-rs0]="01KHKJ02SQM8S8T4B6030F94CV" \
    # [full-rs1]="01KHJ5EQ04XTFG46QCKX3TTAHF" \
    # [full-energy-conserving-rs0]="01KHJ5F1M6YKVZESPZAAVVD6G8" \
    # [full-energy-conserving-rs1]="01KHCXABVNA3TJW0ZT5F4YDDQT" \
)

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT  # so config path is valid no matter where we are running this script

for model in "${!MODELS[@]}"; do
    dataset_id="${MODELS[$model]}"

    for climate in "${!INITIAL_CONDITION_DATASETS[@]}"; do
        co2_concentration=${CO2_CONCENTRATIONS[$climate]}
        initial_condition_path="${INITIAL_CONDITION_DATASETS[$climate]}"
        override="\
            forcing_loader.dataset.overwrite.constant.global_mean_co2=$co2_concentration \
            initial_condition.path=$initial_condition_path \
            initial_condition.start_indices.times=[$INITIAL_CONDITION_TIME] \
        "
        python -m fme.ace.validate_config --config_type inference $CONFIG_PATH --override $override

        job_name="${DATE}-${model}-${climate}-1000-year-equilibrium-climate-inference"
        gantry run \
            --name $job_name \
            --description 'Run inference with ACE' \
            --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
            --workspace ai2/climate-titan \
            --priority urgent \
            --preemptible \
            --cluster ai2/titan \
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
            --system-python \
            --install "pip install --no-deps ." \
            -- python -I -m fme.ace.inference $CONFIG_PATH --override $override
    done
done
