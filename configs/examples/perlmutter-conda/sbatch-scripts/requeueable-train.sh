#!/bin/bash

set -x

# this will manually requeue the job and is called if a timeout signal is received
# see https://docs.nersc.gov/jobs/examples/#preemptible-jobs
preempt_handler()
{
    #place here: commands to run when preempt signal (SIGTERM) arrives from slurm
    kill -TERM ${1} #forward SIGTERM signal to the user application
    #if --requeue was used, slurm will automatically do so here
}
timeout_handler()
{
    kill -TERM ${1}
    scontrol requeue ${SLURM_JOB_ID}
}

TRAIN_CONFIG=${CONFIG_DIR}/train-config.yaml

torchrun --nnodes $SLURM_JOB_NUM_NODES \
 --nproc_per_node $SLURM_GPUS_PER_NODE \
 --rdzv-backend=c10d \
 --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
 -m fme.ace.train ${TRAIN_CONFIG} &

pid=$!
trap "preempt_handler '$pid'" SIGTERM #this catches preempt SIGTERM from slurm
trap "timeout_handler '$pid'" USR1 # this catches timeout USR1 from slurm
if wait $pid; then
    if [ -f ~/.beaker/config.yml ]; then
        echo "Training completed successfully. Uploading artifacts..."
        $CONFIG_DIR/upload-to-beaker.sh $SLURM_JOB_ID
    else
        echo "No beaker config found. Skipping upload."
    fi
else
    echo "Training failed or was interrupted (exit code $?). Skipping upload."
fi
sleep 120
