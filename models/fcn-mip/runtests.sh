#!/bin/bash

set -e
set -x

# Without this the test might stall on selene
export WORLD_SIZE=1

rm -f .coverage


# TODO allow graphcast to run with ensemble_batch_size != 1
coverage run -a bin/inference_ensemble.py tests/configs/graphcast.json

# TODO fix this test
# coverage run -a bin/inference_ensemble.py tests/configs/baseline_afno_26_ifs.json

bin/evaluate_ecwmf.sh 2018-06-01T00:00:00
coverage run -a bin/inference_ensemble.py tests/configs/sfno_73.json
coverage run -a bin/score_ensemble_outputs.py --input_path 'tests/Output.sfno_73ch.Globe.01_01_00_00_00/'
coverage run -a plots/plot_offline_ensemble_scores.py --input_path 'tests/Output.sfno_73ch.Globe.01_01_00_00_00/'

coverage run -a bin/inference_ensemble.py tests/configs/baseline_afno_26.json

# this test will only run on selene
if [[ -f /lustre/fsw/sw_climate_fno/ensemble_init_stats/corr.pth ]]
then
    coverage run -a score-ifs.py --test out.nc
    torchrun --nnodes 1 --nproc_per_node 1 -m fcn_mip.inference_medium_range --test -n 2 graphcast_34ch out.nc
    coverage run -a bin/inference_ensemble.py tests/configs/baseline_afno_26_multi_region.json
fi

coverage run -a bin/inference_ensemble.py tests/configs/hafno.json
coverage report --omit 'test_*.py'

