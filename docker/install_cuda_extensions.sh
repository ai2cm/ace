#!/bin/bash
# Install CUDA extensions for torch-harmonics and Apex GroupNorm.
#
# Installs the CUDA toolkit once, runs both build scripts, then removes
# the toolkit to save image space.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Installing CUDA extensions ==="

# Step 1: Install CUDA toolkit (nvcc) via apt
echo ""
echo "[1/3] Installing CUDA toolkit (nvcc) via apt..."
# Add NVIDIA CUDA apt repository
wget -qO /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
rm /tmp/cuda-keyring.deb
apt-get update
apt-get install -y --no-install-recommends cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-crt-12-8
rm -rf /var/lib/apt/lists/*

export CUDA_HOME=/usr/local/cuda

# Step 2: Build CUDA extensions
echo ""
echo "[2/3] Building CUDA extensions..."
"$SCRIPT_DIR/install_torch_harmonics_cuda.sh"
"$SCRIPT_DIR/install_apex_groupnorm.sh"

# Step 3: Remove CUDA build tools to save space
echo ""
echo "[3/3] Removing CUDA build tools..."
apt-get remove -y cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-crt-12-8
apt-get autoremove -y

echo ""
echo "=== CUDA extensions installation complete ==="
