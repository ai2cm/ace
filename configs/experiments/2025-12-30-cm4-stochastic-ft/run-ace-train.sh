#!/opt/homebrew/bin/bash

set -e

JOB_GROUP="ace2-cm4-atmosphere"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
WANDB_USERNAME=spencerc_ai2
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=4

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

CHECKPOINT_PATH=training_checkpoints/best_ckpt.tar
OVERRIDE="stepper.parameter_init.weights_path=/pre-trained-checkpoint/ckpt.tar"

declare -A FULL=( [cm4_like_am4_random_co2_stochastic-rs0]="01KDF9N4YWG7VC604FKKHYBKB4" [cm4_like_am4_random_co2_stochastic-rs1]="01KD8EBE9XC433J5R59N6NV28M" )
declare -A LIMITED=( [cm4_like_am4_random_co2_stochastic-limited-rs0]="01KDHSHRANH5WACCWQ7AXPW5VR" [cm4_like_am4_random_co2_stochastic-limited-rs1]="01KDFNRZ3K6CD09H9H3HYCS1J1" )

CONFIG_FILENAME="ace-train-config.yaml"
CONFIG_PATH="${SCRIPT_PATH}${CONFIG_FILENAME}"
for name in "${!FULL[@]}"; do
    job_name="2025-12-30-${name}-train"
    pre_trained_checkpoint=${FULL[$name]}
    python -m fme.ace.validate_config --config_type train $CONFIG_PATH --override $OVERRIDE
    gantry run \
        --name $job_name \
        --task-name $job_name \
        --description 'Run ACE training for CM4 atmosphere data' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/climate-titan \
        --priority low \
        --preemptible \
        --cluster ai2/titan \
        --env WANDB_USERNAME=$WANDB_USERNAME \
        --env WANDB_NAME=$job_name \
        --env WANDB_JOB_TYPE=training \
        --env WANDB_RUN_GROUP=$JOB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset spencerc/2025-12-05-CM4-like-AM4-random-CO2-coupled-merged-stats:/atmos_stats \
        --dataset $pre_trained_checkpoint:$CHECKPOINT_PATH:/pre-trained-checkpoint/ckpt.tar \
        --gpus $N_GPUS \
        --shared-memory 400GiB \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_PATH --override $override
done

CONFIG_FILENAME="ace-train-config-limited.yaml"
CONFIG_PATH="${SCRIPT_PATH}${CONFIG_FILENAME}"
for name in "${!LIMITED[@]}"; do
    job_name="2025-12-30-${name}-train"
    pre_trained_checkpoint=${LIMITED[$name]}
    python -m fme.ace.validate_config --config_type train $CONFIG_PATH --override $OVERRIDE
    gantry run \
        --name $job_name \
        --task-name $job_name \
        --description 'Run ACE training for CM4 atmosphere data' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/climate-titan \
        --priority low \
        --preemptible \
        --cluster ai2/titan \
        --env WANDB_USERNAME=$WANDB_USERNAME \
        --env WANDB_NAME=$job_name \
        --env WANDB_JOB_TYPE=training \
        --env WANDB_RUN_GROUP=$JOB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset spencerc/2025-12-05-CM4-like-AM4-random-CO2-coupled-merged-stats:/atmos_stats \
        --dataset $pre_trained_checkpoint:$CHECKPOINT_PATH:/pre-trained-checkpoint/ckpt.tar \
        --gpus $N_GPUS \
        --shared-memory 400GiB \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_PATH --override $override
done