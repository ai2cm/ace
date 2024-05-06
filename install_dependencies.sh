#!/bin/bash

set -e -x

export TORCH_CUDA_ARCH_LIST="compute capability"

python -m pip install -r requirements_except_torch.txt -r fme/dev-requirements.txt
# we need to install torch as a build requirement for later packages
# it is pinned in requirements_outside_of_docker.txt
python -m pip install torch -c requirements_outside_of_docker.txt
python -m pip install -r requirements_outside_of_docker.txt