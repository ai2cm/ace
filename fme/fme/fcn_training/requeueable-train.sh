#!/bin/bash

set -x

YAML_TRAIN_CONFIG=$1
SLEEP=${2:-120}

SCRIPT_DIRECTORY=${0%/*}

# this will manually requeue the job and is called if a timeout signal is received
# see https://docs.nersc.gov/jobs/examples/#preemptible-jobs
timeout_handler()
{
    kill -TERM ${1}
    scontrol requeue ${SLURM_JOB_ID}
}

echo $SLURM_JOB_ID
echo $SLURM_RESTART_COUNT

# run training
torchrun --nproc_per_node $SLURM_GPUS_PER_NODE $SCRIPT_DIRECTORY/train.py --yaml_config $YAML_TRAIN_CONFIG &

pid=$!
trap "timeout_handler '$pid'" USR1 # this catches timeout USR1 from slurm
wait
sleep $SLEEP