#!/bin/bash
# Reinstall torch-harmonics with optimized CUDA kernels
# Tested on: Python 3.11, PyTorch 2.7.1, CUDA 12.8, Ubuntu 22.04
#
# The default pip install of torch-harmonics does not compile custom CUDA
# extensions because the runtime Docker image lacks nvcc. This script
# rebuilds torch-harmonics with CUDA extensions enabled.
# Requires nvcc to be available (see install_cuda_extensions.sh).

set -e

echo "=== Reinstalling torch-harmonics with CUDA extensions ==="

echo ""
echo "Building torch-harmonics with CUDA extensions..."
FORCE_CUDA_EXTENSION=1 \
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0 10.0+PTX" \
    CPLUS_INCLUDE_PATH=/opt/conda/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH \
    pip install --no-build-isolation --no-cache-dir --no-deps --force-reinstall torch-harmonics==0.8.0

echo ""
echo "=== torch-harmonics CUDA installation complete ==="
