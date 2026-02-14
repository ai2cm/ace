#!/opt/homebrew/bin/bash

set -e

CONFIG_FILENAME="multi-step-fine-tune-config-full-beaker.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
WANDB_USERNAME=spencerc_ai2
WANDB_GROUP=ace-shield
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=8
STATS_DATASET=andrep/2026-02-06-vertically-resolved-1deg-c96-shield-ramped-climSST-random-CO2-ensemble-fme-dataset-stats
PRE_TRAINED_WEIGHTS_PATH=/pre-trained-weights/training_checkpoints/best_ckpt.tar
SEED_OFFSET=10

cd $REPO_ROOT  # so config path is valid no matter where we are running this

declare -A PRE_TRAINED_WEIGHTS_DATASETS=( \
    [0]="01KHCDS59F7RBJ1D42WD9HP3XC" \
    [1]="01KHCEF1SBYCZCGDM78N1CJC3H" \
)

for seed in 0 1
do
    job_name="ace-shield-multi-step-fine-tune-full-rs${seed}"
    # Offset seed for fine-tuning so that data shuffling is different than
    # during pre-training, but still follows the same path for a given set
    # random initialization weights.
    fine_tune_seed=$((seed + SEED_OFFSET))
    override="seed=${fine_tune_seed} stepper.parameter_init.weights_path=${PRE_TRAINED_WEIGHTS_PATH}"
    python -m fme.ace.validate_config --config_type train $CONFIG_PATH --override $override
    gantry run \
        --name $job_name \
        --description 'Run ACE training' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/climate-titan \
        --priority urgent \
        --preemptible \
        --cluster ai2/jupiter \
        --env WANDB_NAME=$job_name \
        --env WANDB_USERNAME=$WANDB_USERNAME \
        --env WANDB_JOB_TYPE=training \
        --env WANDB_RUN_GROUP=$WANDB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $STATS_DATASET:/statsdata \
        --dataset ${PRE_TRAINED_WEIGHTS_DATASETS[${seed}]}:/pre-trained-weights \
        --gpus $N_GPUS \
        --shared-memory 400GiB \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_PATH --override $override
done

for seed in 0 1
do
    job_name="ace-shield-multi-step-fine-tune-energy-conserving-full-rs${seed}"
    # Offset seed for fine-tuning so that data shuffling is different than
    # during pre-training, but still follows the same path for a given set
    # random initialization weights.
    fine_tune_seed=$((seed + SEED_OFFSET))
    override="\
        seed=${fine_tune_seed} \
        stepper.parameter_init.weights_path=${PRE_TRAINED_WEIGHTS_PATH} \
        stepper.step.config.corrector.total_energy_budget_correction.method=constant_temperature \
        stepper.step.config.corrector.total_energy_budget_correction.constant_unaccounted_heating=1.13 \
    "
    python -m fme.ace.validate_config --config_type train $CONFIG_PATH --override $override
    gantry run \
        --name $job_name \
        --description 'Run ACE training' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/climate-titan \
        --priority urgent \
        --preemptible \
        --cluster ai2/jupiter \
        --env WANDB_NAME=$job_name \
        --env WANDB_USERNAME=$WANDB_USERNAME \
        --env WANDB_JOB_TYPE=training \
        --env WANDB_RUN_GROUP=$WANDB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $STATS_DATASET:/statsdata \
        --dataset ${PRE_TRAINED_WEIGHTS_DATASETS[${seed}]}:/pre-trained-weights \
        --gpus $N_GPUS \
        --shared-memory 400GiB \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_PATH --override $override
done
