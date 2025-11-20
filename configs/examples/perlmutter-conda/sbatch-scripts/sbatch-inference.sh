#!/bin/bash -l

#SBATCH -A m4492_g
#SBATCH -q regular
#SBATCH -C gpu_hbm40g
#SBATCH -J infer-fme
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 01:00:00
#SBATCH --output=joblogs/%j.out

set -xe

# directory for saving output from training/inference job
export FME_OUTPUT_DIR=${PSCRATCH}/fme-output/${SLURM_JOB_ID}
mkdir -p $FME_OUTPUT_DIR

module load python
conda activate $FME_VENV

# env variables
export WANDB_JOB_TYPE=inference
set +x  # don't print API key to logs
export WANDB_API_KEY=$(cat ~/.config/wandb/api)
set -x

INFERENCE_CONFIG=$CONFIG_DIR/config-inference.yaml

# replace placeholders in config with actual values
sed -i "s|FME_OUTPUT_DIR|${FME_OUTPUT_DIR}|" ${INFERENCE_CONFIG}
sed -i "s|FME_CHECKPOINT_PATH|${FME_CHECKPOINT_PATH}|" ${INFERENCE_CONFIG}
sed -i "s|FME_VALID_DIR|${FME_VALID_DIR}|" ${INFERENCE_CONFIG}

cp -r $CONFIG_DIR $FME_OUTPUT_DIR/job_config

python -u -m fme.ace.evaluator $INFERENCE_CONFIG
