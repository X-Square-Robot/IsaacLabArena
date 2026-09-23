#!/bin/bash
set -euo pipefail

# Script to install CUDA 12.8 for GR00T dependencies (Local Environment Version)
# Adapted from docker/setup/install_cuda.sh for local installation

echo "Installing CUDA 12.8 for GR00T dependencies (Local Environment)"

# Source OS release information
. /etc/os-release

# Detect Ubuntu version and set appropriate CUDA repository
case "$ID" in
  ubuntu)
    case "$VERSION_ID" in
      "20.04") cuda_repo="ubuntu2004";;
      "22.04") cuda_repo="ubuntu2204";;
      "24.04") cuda_repo="ubuntu2404";;
      *) echo "Unsupported Ubuntu $VERSION_ID"; exit 1;;
    esac ;;
  *) echo "Unsupported base OS: $ID"; exit 1 ;;
esac

echo "Detected Ubuntu $VERSION_ID, using repository: $cuda_repo"

# Check if CUDA 12.8 is already installed
if [ -d "/usr/local/cuda-12.8" ]; then
    echo "CUDA 12.8 is already installed at /usr/local/cuda-12.8"
    read -p "Do you want to reinstall? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping CUDA installation"
        exit 0
    fi
fi

# Update package lists and install prerequisites
echo "Installing prerequisites..."
apt-get update
apt-get install -y --no-install-recommends wget gnupg ca-certificates

# Download and install CUDA keyring
echo "Downloading CUDA keyring..."
wget -q https://developer.download.nvidia.com/compute/cuda/repos/${cuda_repo}/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
rm -f cuda-keyring_1.1-1_all.deb

# Download and install CUDA repository pin
echo "Setting up CUDA repository pin..."
wget -q https://developer.download.nvidia.com/compute/cuda/repos/${cuda_repo}/x86_64/cuda-${cuda_repo}.pin
mv cuda-${cuda_repo}.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Update package lists with new CUDA repository
echo "Updating package lists..."
apt-get update

# Install CUDA toolkit 12.8
echo "Installing CUDA Toolkit 12.8 (this may take a while)..."
apt-get install -y --no-install-recommends cuda-toolkit-12-8

# Clean up package cache
echo "Cleaning up..."
apt-get -y autoremove
apt-get clean

# Set CUDA environment variables in user's bashrc
echo ""
echo "Setting up CUDA environment variables..."
CUDA_ENV_LINES="
# CUDA 12.8 Environment Variables (added by install_cuda_local.sh)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:\${LD_LIBRARY_PATH:+\$LD_LIBRARY_PATH:}
export TORCH_CUDA_ARCH_LIST=8.0+PTX
"

if ! grep -q "CUDA 12.8 Environment Variables" ~/.bashrc; then
    echo "$CUDA_ENV_LINES" >> ~/.bashrc
    echo "CUDA environment variables added to ~/.bashrc"
else
    echo "CUDA environment variables already exist in ~/.bashrc"
fi

# Verify installation
if [ -d "/usr/local/cuda-12.8" ]; then
    echo ""
    echo "=========================================="
    echo "CUDA 12.8 installation completed successfully!"
    echo "=========================================="
    echo ""
    echo "To activate CUDA environment variables, run:"
    echo "  source ~/.bashrc"
    echo ""
    echo "Verify installation with:"
    echo "  nvcc --version"
else
    echo "ERROR: CUDA 12.8 installation failed"
    exit 1
fi
