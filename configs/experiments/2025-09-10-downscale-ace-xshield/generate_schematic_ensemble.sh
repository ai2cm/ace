!/bin/bash
# uses the augusta cluster which doesn't have weka access but has GCS access and is
# typically more available than cirrascale clusters

set -e

JOB_NAME="downscale-ace-inference-100km-to-3km-ace-paper-schematic-pnw"

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')

 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

N_NODES=1
NGPU=2

#IMAGE with B200 pytorch installed
IMAGE=01JWJ96JMF89D812JS159VF37N

# Checkpoint at ~
EXISTING_RESULTS_DATASET=01K8RWE83W8BEEAT2KRS94FVCD

ACE_DATASET=01KA4CZZP73VC053HST5HFT3BS
wandb_group=""

run_eval() {
    local ensemble="$1"
    local JOB_NAME_RUN="${JOB_NAME}-ensemble${ensemble}"
    local CONFIG_FILENAME="gen-ace-output-schematic-ic${ensemble}.yaml"
    local CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
    dataset_arg="output_6hourly_predictions_ic${padded}.zarr:/output_6hourly_predictions_ic${padded}.zarr"
    gantry run \
        --name $JOB_NAME_RUN \
        --description 'Run 100km to 3km generation on ACE data' \
        --workspace ai2/ace \
        --priority high \
        --not-preemptible \
        --cluster ai2/titan \
        --beaker-image $IMAGE \
        --env WANDB_USERNAME=$BEAKER_USERNAME \
        --env WANDB_NAME=$JOB_NAME_RUN \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP=$wandb_group \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
	--dataset $EXISTING_RESULTS_DATASET:checkpoints:/checkpoints \
        --dataset $ACE_DATASET:$dataset_arg \
        --weka climate-default:/climate-default \
        --gpus $NGPU \
        --shared-memory 400GiB \
        --budget ai2/climate \
        --system-python \
        --install "pip install --no-deps ." \
        --allow-dirty \
        -- torchrun --nproc_per_node $NGPU -m fme.downscaling.inference $CONFIG_PATH
}

n_ensembles_minus_one=4

for ((i=3; i<=n_ensembles_minus_one; i++)); do
    padded=$(printf "%04d" "$i")   # produces 0000, 0001, ...
    echo "Running ensemble member $padded"
    run_eval "$padded"
done
