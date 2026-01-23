#!/bin/bash
# Install NVIDIA Apex GroupNorm v2 for B200 GPUs (SM100/Blackwell)
# Tested on: Python 3.11, PyTorch 2.7.1, CUDA 12.8, Ubuntu 22.04
#
# Usage: bash install_apex_groupnorm_b200.sh [install_dir]
#   install_dir: Directory to clone Apex into (default: /tmp/nvidia-apex)
#
# This script includes a fix for the missing <tuple> include in the Apex v2 code
# which causes compilation errors with CUDA 12.8 + GCC 11.

set -e

INSTALL_DIR="${1:-/tmp/nvidia-apex}"

echo "=== Installing NVIDIA Apex GroupNorm v2 for B200 ==="
echo "Install directory: $INSTALL_DIR"

# Step 1: Install CUDA toolkit via conda
echo ""
echo "[1/3] Installing CUDA toolkit (nvcc) via conda..."
conda install -y -c nvidia cuda-nvcc=12.8 cuda-toolkit=12.8

# Step 2: Clone Apex
echo ""
echo "[2/3] Cloning NVIDIA Apex..."
if [ -d "$INSTALL_DIR" ]; then
    echo "Directory exists, removing..."
    rm -rf "$INSTALL_DIR"
fi
git clone --depth 1 https://github.com/NVIDIA/apex.git "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Fix missing <tuple> include in group_norm_v2 (required for std::make_tuple, std::get)
echo "Patching missing #include <tuple> in gn_cuda_host_template.cuh..."
sed -i '1a #include <tuple>' apex/contrib/csrc/group_norm_v2/gn_cuda_host_template.cuh

# Step 3: Build and install
echo ""
echo "[3/3] Building Apex with GroupNorm extension only..."
CPLUS_INCLUDE_PATH=/opt/conda/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH \
    APEX_GROUP_NORM=1 \
    pip install -v --no-build-isolation --no-cache-dir ./

# Verify installation
echo ""
echo "=== Verifying installation ==="
python -c "
import torch
from apex.contrib.group_norm import GroupNorm

# Quick test
gn = GroupNorm(num_groups=4, num_channels=16).cuda()
x = torch.randn(2, 16, 8, 8, device='cuda').to(memory_format=torch.channels_last)
y = gn(x)
assert y.shape == x.shape, 'Shape mismatch!'
print('SUCCESS: Apex GroupNorm installed and working!')
print('Usage:')
print('  from apex.contrib.group_norm import GroupNorm')
print('  gn = GroupNorm(num_groups=4, num_channels=64).cuda()')
"

# Remove CUDA build tools to save space
echo "Removing CUDA build tools..."
conda remove -y cuda-nvcc cuda-toolkit --force
conda clean -afy

# Clean up source directory
echo ""
echo "Cleaning up build directory..."
rm -rf "$INSTALL_DIR"

echo ""
echo "=== Installation complete ==="

