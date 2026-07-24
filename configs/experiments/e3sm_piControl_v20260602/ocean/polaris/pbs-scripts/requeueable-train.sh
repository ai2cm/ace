#!/bin/bash

set -x

# This runs once per node (mpiexec --ppn 1). Activate the env on each node so
# torchrun and fme are on PATH.
module use /soft/modulefiles
module load conda
conda activate $FME_VENV

TRAIN_CONFIG=${CONFIG_DIR}/train-config.yaml

# Forward a preemption/timeout signal (PBS sends SIGTERM) to the training
# process so it can checkpoint cleanly. With "#PBS -r y", PBS reruns the job
# after preemption and ACE resumes from the latest checkpoint in experiment_dir.
sig_handler()
{
    kill -TERM ${1}
}

# Static rendezvous: one agent per node with a fixed node rank (from PALS, which
# sets PMI_RANK per mpiexec rank; with --ppn 1 that is the node index). All agents
# connect to MASTER_ADDR:MASTER_PORT. This avoids the dynamic c10d rendezvous,
# which hung indefinitely before Python even started; with a static store a node
# that cannot reach the master fails fast (TCPStore timeout) so PBS ('-r y') can
# requeue instead of silently hanging for the full walltime.
NODE_RANK=${PMI_RANK:-${PALS_RANKID:-0}}
echo "Static rendezvous: NODE_RANK=${NODE_RANK} MASTER=${MASTER_ADDR}:${MASTER_PORT} NNODES=${NNODES} (job ${JOBID})"

# Diagnostics so a startup/rendezvous/NCCL hang is visible in the job log and
# does not stall silently. Override any of these from the environment if needed.
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}

# --no-python so torchrun runs the affinity wrapper, which taskset-binds each
# local rank to its own CPU block before exec'ing `python -m fme.ace.train`.
torchrun --nnodes ${NNODES} \
  --nproc-per-node ${NGPUS_PER_NODE} \
  --node-rank ${NODE_RANK} \
  --master-addr ${MASTER_ADDR} \
  --master-port ${MASTER_PORT} \
  --no-python \
  ${CONFIG_DIR}/set-affinity.sh python -m fme.ace.train ${TRAIN_CONFIG} &

pid=$!
trap "sig_handler '$pid'" SIGTERM
if wait $pid; then
    # only upload once, from the rendezvous (rank 0) node. Use OUTPUT_JOB_ID so
    # resumed jobs upload the original job's output directory.
    if [ "$(hostname)" = "${MASTER_ADDR}" ]; then
        echo "Training completed successfully. Uploading artifacts..."
        $CONFIG_DIR/upload-to-beaker.sh ${OUTPUT_JOB_ID:-$JOBID}
    fi
else
    echo "Training failed or was interrupted (exit code $?). Skipping upload."
fi
sleep 120
