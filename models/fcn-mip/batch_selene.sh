#!/bin/bash
#SBATCH -A devtech
#SBATCH -p luna
#SBATCH -t 00:30:00
#SBATCH -J devtech-e2prep:fcn-mip-ensembles
#SBATCH -N 2
#SBATCH --ntasks-per-node 8
#SBATCH -n 16
#SBATCH --output fcn-mip_16xA100.%j
#SBATCH --dependency singleton

readonly DATA="/lustre/fsw/sw_climate_fno"
readonly CODE="/home/$USER/fcn-mip"

readonly _cont_image=gitlab-master.nvidia.com/earth-2/fcn-mip:23.02.27
readonly _cont_mounts="${DATA}:${DATA}:rw,${CODE}/:/code:rw,/lustre/fsw/nvresearch:/lustre/fsw/nvresearch:ro"

gpus=${SLURM_NTASKS}

srun \
-n ${gpus} \
--container-image="${_cont_image}" \
--container-name="era5_wind" \
--container-mounts="${_cont_mounts}"\
--no-container-entrypoint \
--container-workdir=/code \
bash -c "export LD_LIBRARY_PATH=/usr/local/mpi/lib:$LD_LIBRARY_PATH && python inference_ensemble.py config.afno.in"
