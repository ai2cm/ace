#!/bin/bash -l
# Minimal 2-node distributed connectivity test on the debug queue.
# Submit from the polaris dir:  qsub -V pbs-scripts/pbs-dist-test.sh
# (debug queue: <=2 nodes, <=1h, usually fast turnaround + healthy nodes).
#PBS -A E3SMinput
#PBS -N dist-test
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=00:15:00
#PBS -q debug
#PBS -k doe
#PBS -j oe
#PBS -o joblogs/

set -xe
cd ${PBS_O_WORKDIR}

module use /soft/modulefiles
module load conda
export FME_VENV=${FME_VENV:-/eagle/E3SMinput/elynnwu/fme-env/b1f688621}
conda activate $FME_VENV

export NNODES=$(wc -l < $PBS_NODEFILE)
export NGPUS_PER_NODE=4
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
export MASTER_PORT=29507
export TEST_DIR=${PBS_O_WORKDIR}/pbs-scripts
echo "NNODES=${NNODES} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
cat $PBS_NODEFILE

mpiexec -n ${NNODES} --ppn 1 --cpu-bind none \
  --env FME_VENV=${FME_VENV} \
  --env MASTER_ADDR=${MASTER_ADDR} \
  --env MASTER_PORT=${MASTER_PORT} \
  --env NNODES=${NNODES} \
  --env NGPUS_PER_NODE=${NGPUS_PER_NODE} \
  --env TEST_DIR=${TEST_DIR} \
  ${TEST_DIR}/dist-test-launch.sh
