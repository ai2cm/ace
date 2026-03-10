#!/usr/bin/env bash
set -euo pipefail

H=${FME_DISTRIBUTED_H:-2}
W=${FME_DISTRIBUTED_W:-2}
NP=$((H * W))

export FME_DISTRIBUTED_BACKEND=model
export FME_DISTRIBUTED_H=$H
export FME_DISTRIBUTED_W=$W

# torchrun --standalone --nnodes=1 --nproc_per_node=$NP \
#   -m pytest fme/core/distributed/parallel_tests/test_spatial.py "$@"

torchrun --standalone --nnodes=1 --nproc_per_node=$NP \
  -m pytest fme/core/distributed/parallel_tests/test_backward_step.py::test_spatial_parallel_backward_step "$@"

