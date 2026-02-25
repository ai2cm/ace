#!/opt/homebrew/bin/bash

set -e

DATE="2026-02-23"
CONFIG_FILENAME="ace-1xCO2-ensemble-inference-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
WANDB_USERNAME=spencerc_ai2
REPO_ROOT=$(git rev-parse --show-toplevel)

CHECKPOINT_PATH=training_checkpoints/best_inference_ckpt.tar

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

declare -A MODELS=( \
    # [published-baseline-rs3]="01J4BR6J5AW32ZDQ77VZ60P4KT" \
    [residual-predicting-energy-conserving-multi-call-rs0]="01JQNAQC4PJDX0765PEDEZ15HR" \
    # [no-random-co2-rs0]="01KHGDAMB2BDZQS8JFF65A2YDR" \
    # [no-random-co2-rs1]="01KH4SDCYN1NF2RP2JXZS0WZ1Y" \
    # [no-random-co2-energy-conserving-rs0]="01KHGDA8TVGP9JKWVJ1N0SMHCN" \
    # [no-random-co2-energy-conserving-rs1]="01KH4SDT1Q5246GZ307W8AW4M3" \
    # [full-rs0]="01KHKJ02SQM8S8T4B6030F94CV" \
    # [full-rs1]="01KHJ5EQ04XTFG46QCKX3TTAHF" \
    # [full-energy-conserving-rs0]="01KHJ5F1M6YKVZESPZAAVVD6G8" \
    # [full-energy-conserving-rs1]="01KHCXABVNA3TJW0ZT5F4YDDQT" \
)

OVERRIDE="n_forward_steps=28"
for name in "${!MODELS[@]}"; do
    python -m fme.ace.validate_config --config_type inference $CONFIG_PATH --override $OVERRIDE
    job_name="${DATE}-${name}-seven-day-1xCO2-ensemble-inference"
    existing_results_dataset=${MODELS[$name]}
    gantry run \
        --name $job_name \
        --description 'Run seven-day ACE 1xCO2 ensemble inference' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        --preemptible \
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
        --dataset $existing_results_dataset:$CHECKPOINT_PATH:/ckpt.tar \
        --gpus 1 \
        --shared-memory 20GiB \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- python -I -m fme.ace.inference $CONFIG_PATH --override $OVERRIDE
done

OVERRIDE="n_forward_steps=28 forcing_loader.dataset.overwrite.constant.global_mean_co2=0.0014537"  # Abrupt 4xCO2
for name in "${!MODELS[@]}"; do
    python -m fme.ace.validate_config --config_type inference $CONFIG_PATH --override $OVERRIDE
    job_name="${DATE}-${name}-seven-day-abrupt-4xCO2-ensemble-inference"
    existing_results_dataset=${MODELS[$name]}
    gantry run \
        --name $job_name \
        --description 'Run seven-day ACE abrupt 4xCO2 ensemble inference' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        --preemptible \
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
        --dataset $existing_results_dataset:$CHECKPOINT_PATH:/ckpt.tar \
        --gpus 1 \
        --shared-memory 20GiB \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- python -I -m fme.ace.inference $CONFIG_PATH --override $OVERRIDE
done
