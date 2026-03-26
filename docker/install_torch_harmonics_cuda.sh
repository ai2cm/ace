#!/bin/bash
# Reinstall torch-harmonics with optimized CUDA kernels
# Tested on: Python 3.11, PyTorch 2.7.1, CUDA 12.8, Ubuntu 22.04
#
# The default pip install of torch-harmonics does not compile custom CUDA
# extensions because the runtime Docker image lacks nvcc. This script
# temporarily installs the CUDA toolkit, rebuilds torch-harmonics with
# CUDA extensions enabled, then removes the toolkit to save space.

set -e

echo "=== Reinstalling torch-harmonics with CUDA extensions ==="

# Step 1: Install CUDA toolkit via conda
echo ""
echo "[1/3] Installing CUDA toolkit (nvcc) via conda..."
conda install -y -c nvidia cuda-nvcc=12.8 cuda-toolkit=12.8

# Step 2: Reinstall torch-harmonics with CUDA extensions
echo ""
echo "[2/3] Building torch-harmonics with CUDA extensions..."
FORCE_CUDA_EXTENSION=1 \
    TORCH_CUDA_ARCH_LIST="7.5 9.0 10.0+PTX" \
    CPLUS_INCLUDE_PATH=/opt/conda/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH \
    pip install --no-build-isolation --no-cache-dir --no-deps --force-reinstall torch-harmonics==0.8.0

# Step 3: Remove CUDA build tools to save space
echo ""
echo "[3/3] Removing CUDA build tools..."
conda remove -y cuda-nvcc cuda-toolkit --force
conda clean -afy

echo ""
echo "=== torch-harmonics CUDA installation complete ==="
