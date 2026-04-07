#!/bin/bash
# Install NVIDIA Apex GroupNorm
# Tested on: Python 3.11, PyTorch 2.7.1, CUDA 12.8, Ubuntu 22.04
#
# This script includes a fix for the missing <tuple> include in the Apex v2 code
# which causes compilation errors with CUDA 12.8 + GCC 11.

set -e

INSTALL_DIR="${1:-/tmp/nvidia-apex}"

echo "=== Installing NVIDIA Apex GroupNorm ==="
echo "Install directory: $INSTALL_DIR"

# Step 1: Clone Apex
echo ""
echo "[1/2] Cloning NVIDIA Apex..."
if [ -d "$INSTALL_DIR" ]; then
    echo "Directory exists, removing..."
    rm -rf "$INSTALL_DIR"
fi
APEX_COMMIT="3c57b14c042a89957de51a1f476472fa6ec1e46a"
git init "$INSTALL_DIR"
cd "$INSTALL_DIR"
git fetch --depth 1 https://github.com/NVIDIA/apex.git "$APEX_COMMIT"
git checkout FETCH_HEAD

# Fix missing <tuple> include in group_norm_v2 (required for std::make_tuple, std::get)
sed -i '1a #include <tuple>' apex/contrib/csrc/group_norm_v2/gn_cuda_host_template.cuh

# Step 2: Build and install
echo ""
echo "[2/2] Building Apex with GroupNorm extension only..."
CPLUS_INCLUDE_PATH=/opt/conda/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH \
    APEX_GROUP_NORM=1 \
    pip install -v --no-build-isolation --no-cache-dir ./

# Clean up source directory
echo ""
echo "Cleaning up build directory..."
rm -rf "$INSTALL_DIR"

echo ""
echo "=== Installation complete ==="

