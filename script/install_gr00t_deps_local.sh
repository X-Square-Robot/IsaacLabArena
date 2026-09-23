#!/bin/bash
set -euo pipefail

# Script to install GR00T policy dependencies (Local Environment Version)
# Adapted from docker/setup/install_gr00t_deps.sh for local installation
# This script assumes an activated conda or uv environment.

# Check required environment variables
if [ -z "${GROOT_DEPS_GROUP:-}" ]; then
    echo "ERROR: GROOT_DEPS_GROUP environment variable is not set"
    echo "Usage: GROOT_DEPS_GROUP=base bash $0"
    exit 1
fi

if [ -z "${ARENA_DIR:-}" ]; then
    echo "ERROR: ARENA_DIR environment variable is not set"
    echo "This script should be called from setup_arena_local.sh"
    exit 1
fi

echo "Installing GR00T with dependency group: $GROOT_DEPS_GROUP"
echo "Arena directory: $ARENA_DIR"

# Resolve the active Python backend. Callers pass MANAENV_ENV_BACKEND; direct
# invocations can infer it from the standard activation variables.
ENV_BACKEND="${MANAENV_ENV_BACKEND:-}"
if [ -z "$ENV_BACKEND" ]; then
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        ENV_BACKEND="uv"
    elif [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
        ENV_BACKEND="conda"
    fi
fi

if [ "$ENV_BACKEND" = "uv" ]; then
    if [ -z "${VIRTUAL_ENV:-}" ] || [ ! -x "$VIRTUAL_ENV/bin/python" ]; then
        echo "ERROR: An activated uv environment is required"
        exit 1
    fi
    echo "Using uv environment: $VIRTUAL_ENV"
elif [ "$ENV_BACKEND" = "conda" ]; then
    if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
        echo "ERROR: No active conda environment detected"
        exit 1
    fi
    echo "Using conda environment: $CONDA_DEFAULT_ENV"
else
    echo "ERROR: No active conda or uv environment detected"
    exit 1
fi

target_python="$(command -v python)"
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is unavailable; bootstrapping it with the target Python"
    "$target_python" -m pip install uv
fi

pip_install() {
    uv pip install --python "$target_python" "$@"
}

echo "Python: $(which python)"

# Set CUDA environment variables for GR00T installation. Preserve an explicit
# CUDA_HOME; otherwise derive it from the compiler selected by the install shell.
if [ -z "${CUDA_HOME:-}" ]; then
    if command -v nvcc >/dev/null 2>&1; then
        nvcc_path="$(command -v nvcc)"
        if command -v readlink >/dev/null 2>&1; then
            nvcc_path="$(readlink -f "$nvcc_path")"
        fi
        CUDA_HOME="$(dirname "$(dirname "$nvcc_path")")"
    else
        CUDA_HOME=/usr/local/cuda-12.8
    fi
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:${PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=8.0+PTX

echo ""
echo "CUDA environment variables:"
echo "  CUDA_HOME=$CUDA_HOME"
echo "  PATH=$PATH"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

# Check the actual compiler, not only the installation directory name.
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo ""
    echo "ERROR: CUDA Toolkit compiler not found at $CUDA_HOME/bin/nvcc"
    echo "Install and configure CUDA Toolkit 12.8 yourself, then rerun the ManaEnv installer."
    echo "Verify before retrying: nvcc --version"
    exit 1
fi
if ! nvcc_info="$("$CUDA_HOME/bin/nvcc" --version 2>&1)" || [[ "$nvcc_info" != *"release 12.8"* ]]; then
    echo ""
    echo "ERROR: CUDA Toolkit 12.8 is required; detected compiler: $CUDA_HOME/bin/nvcc"
    echo "$nvcc_info"
    echo "Install and configure CUDA Toolkit 12.8 yourself, then rerun the ManaEnv installer."
    echo "Verify before retrying: nvcc --version"
    exit 1
fi

# Installing system-level media libraries
echo ""
echo "Installing system-level media libraries..."
if command -v apt-get &>/dev/null; then
    sudo apt-get update && sudo apt-get install -y ffmpeg
    echo "ffmpeg installed successfully"
else
    echo "WARNING: apt-get not found, skipping ffmpeg installation"
fi

# Upgrade packaging tools to avoid setuptools issues
echo ""
echo "Upgrading packaging tools..."
pip_install --upgrade setuptools packaging wheel

# Verify Isaac-GR00T directory
GROOT_DIR="$ARENA_DIR/submodules/Isaac-GR00T"
if [ ! -d "$GROOT_DIR" ]; then
    echo ""
    echo "ERROR: Isaac-GR00T directory not found: $GROOT_DIR"
    echo "Please prepare it through ManaEnv install.sh Step 1 first"
    exit 1
fi

# Install GR00T with the specified dependency group
echo ""
echo "Installing Isaac-GR00T with dependency group: $GROOT_DEPS_GROUP"
echo "From directory: $GROOT_DIR"
pip_install --no-build-isolation --use-pep517 -e "$GROOT_DIR[$GROOT_DEPS_GROUP]"

# Install flash-attn (specific version for compatibility)
if [ "${BUILD_FLASH_ATTN:-false}" = "true" ]; then
    echo ""
    echo "Building flash-attn from source..."
    echo "NOTE: This may take a long time and requires CUDA for compilation"
    pip_install --no-build-isolation --use-pep517 flash-attn==2.7.1.post4
fi

# Verify installation
echo ""
echo "Verifying GR00T installation..."
if python -c "import gr00t; print(f'GR00T version: {gr00t.__version__}')" 2>/dev/null; then
    echo ""
    echo "=========================================="
    echo "GR00T dependencies installation completed successfully!"
    echo "=========================================="
    echo ""
    echo "Verify installation with:"
    echo "  python -c \"import gr00t; print('GR00T OK')\""
else
    echo ""
    echo "WARNING: GR00T import failed"
    echo "This might be expected if Isaac Sim environment is required"
fi
