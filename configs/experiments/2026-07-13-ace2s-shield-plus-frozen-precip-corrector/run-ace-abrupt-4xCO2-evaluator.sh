#!/opt/homebrew/bin/bash

set -e

DATE=$(date +%Y-%m-%d)
CONFIG_FILENAME="ace-abrupt-4xCO2-evaluator-config.yaml"
BEAKER_IMAGE=jeremym/fme-deps-only-5039277ac
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
WANDB_USERNAME=spencerc_ai2
REPO_ROOT=$(git rev-parse --show-toplevel)
BEAKER_IMAGE="$(cat $REPO_ROOT/latest_deps_only_image.txt)"

MAIN_COMMIT=edcebd0ebdc8ea4f9eec589270879425f06b7feb
FROZEN_PRECIP_CORRECTOR_COMMIT=fa9b0e3f94030a6999ff6691bf7f57d90227e7d9
CHECKPOINT_PATH=training_checkpoints/best_inference_ckpt.tar

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

CONFIG_B64=$(base64 < "$CONFIG_PATH" | tr -d '\n')

declare -A MODELS=( \
    # [published-baseline-rs3]="01J4BR6J5AW32ZDQ77VZ60P4KT" \
    # [no-random-co2-rs0]="01KHGDAMB2BDZQS8JFF65A2YDR" \
    # [no-random-co2-rs1]="01KH4SDCYN1NF2RP2JXZS0WZ1Y" \
    # [no-random-co2-energy-conserving-rs0]="01KHGDA8TVGP9JKWVJ1N0SMHCN" \
    # [no-random-co2-energy-conserving-rs1]="01KH4SDT1Q5246GZ307W8AW4M3" \
    # [full-rs0]="01KHKJ02SQM8S8T4B6030F94CV" \
    # [full-rs1]="01KHJ5EQ04XTFG46QCKX3TTAHF" \
    [full-energy-conserving-rs0]="01KHJ5F1M6YKVZESPZAAVVD6G8" \
    # [full-energy-conserving-rs1]="01KHCXABVNA3TJW0ZT5F4YDDQT" \
)

for name in "${!MODELS[@]}"; do
    python -m fme.ace.validate_config --config_type evaluator $CONFIG_PATH
    job_name="${DATE}-${name}-abrupt-4xCO2-evaluator-main"
    existing_results_dataset=${MODELS[$name]}
    gantry run \
        --remote https://github.com/ai2cm/ace \
        --ref ${MAIN_COMMIT} \
        --name $job_name \
        --description 'Run ACE abrupt 4xCO2 evaluator' \
        --beaker-image "${BEAKER_IMAGE}" \
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
        --dataset $existing_results_dataset:$CHECKPOINT_PATH:/ckpt.tar \
        --gpus 1 \
        --shared-memory 20GiB \
        --weka climate-default:/climate-default \
        --system-python \
        --install "pip install --no-deps ." \
        -- bash -c "\
            echo '${CONFIG_B64}' | base64 -d > /tmp/config.yaml \
            && \
            python -I -m fme.ace.evaluator /tmp/config.yaml \
        "

    job_name="${DATE}-${name}-abrupt-4xCO2-evaluator-frozen-precip-corrector"
    existing_results_dataset=${MODELS[$name]}
    gantry run \
        --remote https://github.com/ai2cm/ace \
        --ref ${FROZEN_PRECIP_CORRECTOR_COMMIT} \
        --name $job_name \
        --description 'Run ACE abrupt 4xCO2 evaluator' \
        --beaker-image "${BEAKER_IMAGE}" \
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
        --dataset $existing_results_dataset:$CHECKPOINT_PATH:/ckpt.tar \
        --gpus 1 \
        --shared-memory 20GiB \
        --weka climate-default:/climate-default \
        --system-python \
        --install "pip install --no-deps ." \
        -- bash -c "\
            echo '${CONFIG_B64}' | base64 -d > /tmp/config.yaml \
            && \
            python -I -m fme.ace.evaluator /tmp/config.yaml \
        "
done
