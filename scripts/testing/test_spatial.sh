#!/usr/bin/env bash
set -euo pipefail
set -x 

H_ARG=${1:-}
W_ARG=${2:-}

H=${H_ARG:-${FME_DISTRIBUTED_H:-2}}
W=${W_ARG:-${FME_DISTRIBUTED_W:-2}}

NP=$((H * W))

dir=fme/core/distributed/parallel_tests
# dir=fme/ace/stepper/
tests=test_backward_step.py::test_spatial_parallel_backward_step 
# tests=test_step.py::test_stepper_backward_with_optimization
# tests=test_single_module_csfno.py::test_stepper_train_on_batch_with_optimization_regression
pytest_cmd="pytest -s $dir/$tests"
file="testdata/backward_step_baseline.pt"
# file="testdata/csfno_stepper_backward_with_opt_baseline.pt"

if [ -f "$dir/$file" ]; then
  rm "$dir/$file"
fi

$pytest_cmd

export FME_DISTRIBUTED_BACKEND=model
export FME_DISTRIBUTED_H=$H
export FME_DISTRIBUTED_W=$W
torchrun --standalone --nnodes=1 --nproc-per-node=$NP -m $pytest_cmd

