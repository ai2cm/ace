#!/opt/homebrew/bin/bash

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
BEAKER_USERNAME=spencerc_ai2
REPO_ROOT=$(git rev-parse --show-toplevel)
CHECKPOINT_PATH=training_checkpoints/best_inference_ckpt.tar

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

declare -A MODELS=( \
    [cm4_like_am4_random_co2_ft_ec-rs0]="01KCPP22XQ22872WFMV99K6GDV" \
    [cm4_like_am4_random_co2_ft_ec-rs1]="01KE0FGWC8SSE8SMXV9RAWXS12" \
    [cm4_like_am4_random_co2_ft_ec-rs2]="01KE0HNNDTX0T6S02XQ1V4QX2M" \
)

case=out-of-sample-1pctCO2
config_path=${SCRIPT_PATH}/ace-out-of-sample-1pctCO2-evaluator.yaml
python -m fme.ace.validate_config --config_type evaluator ${config_path}
for name in "${!MODELS[@]}"; do
    job_name="2026-01-13-$case-evaluator-$name"
    existing_results_dataset=${MODELS[$name]}
    gantry run \
            --name $job_name \
            --description 'Run ACE random CO2 evaluator' \
            --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
            --workspace ai2/ace \
            --priority high \
            --preemptible \
            --cluster ai2/ceres \
            --cluster ai2/jupiter \
            --cluster ai2/titan \
            --env WANDB_USERNAME=$BEAKER_USERNAME \
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
            -- python -I -m fme.ace.evaluator ${config_path}
done
