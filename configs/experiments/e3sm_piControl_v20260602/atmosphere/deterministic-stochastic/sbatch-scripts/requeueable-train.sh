#!/bin/bash

set -x

preempt_handler()
{
    kill -TERM "${1}"
}

timeout_handler()
{
    kill -TERM "${1}"
    scontrol requeue "${SLURM_JOB_ID}"
}

TRAIN_CONFIG=${CONFIG_DIR}/train-config.yaml
TRAIN_ARGS=("$TRAIN_CONFIG")

if [[ -n "${FME_OVERRIDE_ARGS:-}" ]]; then
    read -r -a OVERRIDE_ARRAY <<< "$FME_OVERRIDE_ARGS"
    TRAIN_ARGS+=("--override" "${OVERRIDE_ARRAY[@]}")
fi

torchrun --nnodes "$SLURM_JOB_NUM_NODES" \
 --nproc_per_node "$SLURM_GPUS_PER_NODE" \
 --rdzv-backend=c10d \
 --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
 -m fme.ace.train "${TRAIN_ARGS[@]}" &

pid=$!
trap "preempt_handler '$pid'" SIGTERM
trap "timeout_handler '$pid'" USR1
wait "$pid"
sleep 120
