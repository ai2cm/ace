#!/bin/bash
# Install CUDA extensions for torch-harmonics and Apex GroupNorm.
#
# Installs the CUDA toolkit once, runs both build scripts, then removes
# the toolkit to save image space.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Installing CUDA extensions ==="

# Step 1: Install CUDA toolkit via apt
echo ""
echo "[1/3] Installing CUDA toolkit (nvcc) via apt..."
apt-get update && apt-get install -y --no-install-recommends \
    cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-crt-12-8
rm -rf /var/lib/apt/lists/*

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
