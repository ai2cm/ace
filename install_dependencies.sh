#!/bin/bash

set -e -x

export TORCH_CUDA_ARCH_LIST="compute capability"

python -m pip install -r requirements_except_torch.txt -r fme/dev-requirements.txt
python -m pip install --no-deps -r requirements_no_deps.txt
python -m pip install -r requirements_outside_of_docker.txt