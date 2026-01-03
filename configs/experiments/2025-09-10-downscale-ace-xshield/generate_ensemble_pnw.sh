# uses the augusta cluster which doesn't have weka access but has GCS access and is
# typically more available than cirrascale clusters

set -e

JOB_NAME="downscale-ace-100km-to-3km-10-year-pnw-no-churn"

CONFIG_FILENAME="gen-deterministic-ace-output-pnw-1-year-ic0000.yaml"

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME

 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

NGPU=2
IMAGE="$(cat $REPO_ROOT/latest_deps_only_image.txt)"

EXISTING_RESULTS_DATASET=01K8RWE83W8BEEAT2KRS94FVCD

wandb_group=""

run_eval() {
    local ensemble="$1"
    local JOB_NAME_RUN="${JOB_NAME}-ic${ensemble}"
    local CONFIG_FILENAME="gen-ace-output-pnw-ic${ensemble}.yaml"
    local CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
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
        --weka climate-default:/climate-default \
        --gpus $NGPU \
        --shared-memory 400GiB \
        --budget ai2/climate \
        --system-python \
        --install "pip install --no-deps ." \
        --allow-dirty \
       -- torchrun --nproc_per_node $NGPU -m fme.downscaling.inference $CONFIG_PATH
}

n_ensembles_minus_one=9

for ((i=0; i<=n_ensembles_minus_one; i++)); do
    padded=$(printf "%04d" "$i")   # produces 0000, 0001, ...
    echo "Running ensemble member $padded"
    run_eval "$padded"
done