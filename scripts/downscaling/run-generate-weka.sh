#!/bin/bash
# uses the augusta cluster which doesn't have weka access but has GCS access and is
# typically more available than cirrascale clusters

set -e

JOB_NAME="conus-generate-25km-to-3km-pr-winds-july-oct-2021"

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')

 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

N_GPUS=8
N_NODES=5

#IMAGE with B200 pytorch installed
IMAGE=01JWJ96JMF89D812JS159VF37N

TORCHRUN_RDZV_ARGS="--rdzv_id 101 --rdzv_backend static --node_rank \$BEAKER_REPLICA_RANK --rdzv_endpoint \$BEAKER_LEADER_REPLICA_HOSTNAME:44444"
GANTRY_MULTINODE_ARGS=(
    "--host-networking
    --replicas $N_NODES
    --leader-selection
    --propagate-failure
    --propagate-preemption
    --synchronized-start-timeout 15m
    --retries 1
    "
)

command="torchrun --nnodes $N_NODES --nproc_per_node $N_GPUS $TORCHRUN_RDZV_ARGS $SCRIPT_PATH/save_generation.py"

gantry run \
    --name $JOB_NAME \
    --description 'Run 25km to 3km generation CONUS' \
    --workspace ai2/downscaling \
    --priority normal \
    --preemptible \
    --cluster ai2/titan-cirrascale \
    --beaker-image $IMAGE \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP= \
    --env NCCL_SOCKET_IFNAME=ib \
    --env NCCL_IB_HCA="^=mlx5_bond_0" \
    --env NCCL_DEBUG=ERROR \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --weka climate-default:/climate-default \
    --gpus $N_GPUS \
    --shared-memory 400GiB \
    --budget ai2/climate \
    --no-conda \
    $GANTRY_MULTINODE_ARGS \
    --install "pip install --no-deps ." \
    -- /bin/bash -c "$command"
