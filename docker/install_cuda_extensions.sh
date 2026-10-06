#!/bin/bash
# Install CUDA extensions for torch-harmonics and Apex GroupNorm.
#
# The PyTorch runtime base image is built on plain Ubuntu and gets CUDA from pip
# wheels, so it has no nvcc. This installs the CUDA toolkit from NVIDIA's apt
# repository, builds the extensions against it, then purges it again so only the
# built extensions are left behind.
#
# Must be run as a single Docker RUN step, otherwise the toolkit is captured in
# an image layer and purging it later does not reclaim the space.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Keep in sync with the CUDA version of the base image in docker/Dockerfile.
CUDA_APT_SUFFIX=12-8
CUDA_INSTALL_DIR=/usr/local/cuda-12.8

echo "=== Installing CUDA extensions ==="

# Step 1: Install CUDA toolkit via apt
echo ""
echo "[1/3] Installing CUDA toolkit (nvcc) via apt..."
# Record the installed packages first, so that step 3 purges exactly what was
# added here, down to the keyring that registers NVIDIA's apt repository. The
# base image installs no CUDA apt packages, but diffing also keeps the purge
# from reaching anything that was already present.
PACKAGES_BEFORE=$(mktemp)
PACKAGES_ADDED=$(mktemp)
dpkg-query -W -f='${Package}\n' | sort > "${PACKAGES_BEFORE}"

KEYRING_DIR=$(mktemp -d)
curl -fsSL -o "${KEYRING_DIR}/cuda-keyring.deb" \
    "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
dpkg -i "${KEYRING_DIR}/cuda-keyring.deb"
rm -rf "${KEYRING_DIR}"
apt-get update -y

apt-get install -y --no-install-recommends "cuda-toolkit-${CUDA_APT_SUFFIX}"

dpkg-query -W -f='${Package}\n' | sort | comm -13 "${PACKAGES_BEFORE}" - \
    > "${PACKAGES_ADDED}"

export CUDA_HOME="${CUDA_INSTALL_DIR}"
export PATH="${CUDA_HOME}/bin:${PATH}"

if ! command -v nvcc > /dev/null; then
    echo "nvcc not found on PATH after installing the CUDA toolkit" >&2
    exit 1
fi
nvcc --version

# Step 2: Build CUDA extensions
echo ""
echo "[2/3] Building CUDA extensions..."
"$SCRIPT_DIR/install_torch_harmonics_cuda.sh"
"$SCRIPT_DIR/install_apex_groupnorm.sh"

# Step 3: Remove CUDA build tools to save space
echo ""
echo "[3/3] Removing CUDA build tools..."
xargs -r apt-get purge -y < "${PACKAGES_ADDED}"
rm -f "${PACKAGES_BEFORE}" "${PACKAGES_ADDED}"
apt-get clean
rm -rf /var/lib/apt/lists/*
# dpkg only removes the files it installed, so drop anything the builds left in
# the toolkit directory. The /usr/local/cuda symlink is an update-alternatives
# link that the toolkit's postrm removes, but clean it up if it is left dangling.
rm -rf "${CUDA_INSTALL_DIR}"
if [ -L /usr/local/cuda ] && [ ! -e /usr/local/cuda ]; then
    rm -f /usr/local/cuda
fi

echo ""
echo "=== CUDA extensions installation complete ==="
