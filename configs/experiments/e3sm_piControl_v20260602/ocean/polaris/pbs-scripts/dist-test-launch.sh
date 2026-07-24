#!/bin/bash
# Per-node launcher for the distributed connectivity test (run via mpiexec --ppn 1).
set -x

module use /soft/modulefiles
module load conda
conda activate $FME_VENV

# Static rendezvous: node rank from PALS (PMI_RANK per mpiexec rank, --ppn 1).
NODE_RANK=${PMI_RANK:-${PALS_RANKID:-0}}
echo "dist-test: host=$(hostname) NODE_RANK=${NODE_RANK} MASTER=${MASTER_ADDR}:${MASTER_PORT} NNODES=${NNODES}"

export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}

torchrun --nnodes ${NNODES} \
  --nproc-per-node ${NGPUS_PER_NODE} \
  --node-rank ${NODE_RANK} \
  --master-addr ${MASTER_ADDR} \
  --master-port ${MASTER_PORT} \
  ${TEST_DIR}/dist-test.py
