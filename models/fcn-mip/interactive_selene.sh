#!/bin/bash

readonly DATA="/lustre/fsw/sw_climate_fno"
readonly CODE="/home/$USER/"

readonly _cont_image=gitlab-master.nvidia.com/earth-2/fcn-mip:23.04.04
readonly _cont_mounts="${DATA}:${DATA}:rw,${CODE}/:/code:rw,/lustre/fsw/nvresearch:/lustre/fsw/nvresearch:ro"
jobName=devtech-e2prep:vscode-$USER

srun \
-A devtech \
-p interactive \
-t 02:00:00 \
-N 1 \
--ntasks-per-node=1 \
--job-name $jobName \
--container-image="${_cont_image}" \
--container-name="era5_wind" \
--container-mounts="${_cont_mounts}"\
--no-container-entrypoint \
--container-workdir=/code \
--pty bash
