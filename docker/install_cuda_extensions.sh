#!/bin/bash
# Install CUDA extensions for torch-harmonics and Apex GroupNorm.
#
# Requires nvcc, which the `devel` flavor of the PyTorch base image provides
# (see FLAVOR in docker/Dockerfile).

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Installing CUDA extensions ==="

if ! command -v nvcc > /dev/null; then
    echo "nvcc not found; build the image with --build-arg FLAVOR=devel" >&2
    exit 1
fi

"$SCRIPT_DIR/install_torch_harmonics_cuda.sh"
"$SCRIPT_DIR/install_apex_groupnorm.sh"

echo ""
echo "=== CUDA extensions installation complete ==="
