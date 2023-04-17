# Ensemble Hindcasts

This is a selene workflow for running a hindcasts from 2000 to 2015. Each
hindcast is the ensemble mean for an ensemble of 8 initial conditions. initial
conditions are taken from each week of the year. See [hindcast.py](hindcast.py)
for the configuration information.

This workflow is parallelized as follows:
- each year is a separate 1 node slurm job
- The ensembles are parallelized across the GPUs of each node.

It takes around 1 hour to run this hindcast.

## Quickstart

To submit hindcasts for 2000 to 2015

	./cli.sh submit

To view the available ICs

	./cli.sh list

The data are saved at paths like this

	/lustre/fsw/sw_climate_fno/nbrenowitz/hindcast/<fcn model>/<initial-time>.nc
