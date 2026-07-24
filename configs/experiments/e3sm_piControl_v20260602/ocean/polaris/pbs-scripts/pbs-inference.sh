#!/bin/bash -l

# Mandatory at ALCF: project, walltime, filesystems.
#PBS -A E3SMinput
#PBS -N infer-fme
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=01:00:00
#PBS -q preemptable
#PBS -k doe
#PBS -j oe
#PBS -o joblogs/

set -xe

cd ${PBS_O_WORKDIR}

JOBID=${PBS_JOBID%%.*}

FME_SCRATCH=${FME_SCRATCH:-/eagle/E3SMinput/elynnwu}

# directory for saving output from the inference job
export FME_OUTPUT_DIR=${FME_SCRATCH}/fme-output/${JOBID}
mkdir -p $FME_OUTPUT_DIR

module use /soft/modulefiles
module load conda
conda activate $FME_VENV

# env variables
export WANDB_JOB_TYPE=inference
set +x  # don't print API key to logs
# Use WANDB_API_KEY from the environment (e.g. exported in ~/.bashrc) or fall
# back to `wandb login` credentials in ~/.netrc. Only read the legacy key file
# if it exists, and never clobber an already-set key.
if [ -z "${WANDB_API_KEY}" ] && [ -f ~/.config/wandb/api ]; then
  export WANDB_API_KEY=$(cat ~/.config/wandb/api)
fi
set -x

INFERENCE_CONFIG=$CONFIG_DIR/config-inference.yaml

# replace placeholders in config with actual values
sed -i "s|FME_OUTPUT_DIR|${FME_OUTPUT_DIR}|" ${INFERENCE_CONFIG}
sed -i "s|FME_CHECKPOINT_PATH|${FME_CHECKPOINT_PATH}|" ${INFERENCE_CONFIG}
sed -i "s|FME_VALID_DIR|${FME_VALID_DIR}|" ${INFERENCE_CONFIG}

cp -r $CONFIG_DIR $FME_OUTPUT_DIR/job_config

# Inference uses a single GPU on one node.
python -u -m fme.ace.evaluator $INFERENCE_CONFIG
