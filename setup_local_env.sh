#! /bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

# Ensure the script is sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "❌ This script must be sourced, not executed. Use:"
    echo "    source $0"
    exit 1
fi

# Get the repository root directory
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)

# Probably not necessary, but just in case, we do an lfs pull
echo "Ensuring Git LFS files are pulled..."
git lfs pull
if [[ $? -ne 0 ]]; then
    echo "❌ Git LFS pull failed. Exiting."
    return 1
fi

# Check for Rust toolchain (required for utils_rs maturin build)
if ! command -v cargo &> /dev/null; then
    echo "⚠️  Rust toolchain (cargo) not found. It is required for building utils_rs."
    read -p "Would you like to install it via rustup? [y/N] " -r
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        echo "Installing Rust toolchain..."
        curl --proto '=https' --tlsv1.2 -sSf --connect-timeout 10 --max-time 300 https://sh.rustup.rs | sh -s -- -y
        if [[ $? -ne 0 ]]; then
            echo "❌ Failed to install Rust toolchain. Exiting."
            return 1
        fi
        source "$HOME/.cargo/env"
        if ! command -v cargo &> /dev/null; then
            echo "❌ cargo not found in PATH after sourcing ~/.cargo/env. Exiting."
            return 1
        fi
        echo "✅ Rust toolchain installed successfully."
    else
        echo "❌ Rust toolchain is required. Exiting."
        return 1
    fi
fi

# Setup GRPC
echo "Setting up GRPC..."
pushd "${REPO_ROOT}/src/grpc" > /dev/null
uv run compile-protos
if [[ $? -ne 0 ]]; then
    echo "❌ Failed to compile protobufs. Exiting."
    popd > /dev/null
    return 1
fi
popd > /dev/null

# refresh utils_rs package (it doesn't auto-update because it's a compiled extension)
uv pip install --force-reinstall -e "${REPO_ROOT}/src/utils_rs"


# Download vavam models if not already present
VAVAM_DIR="${REPO_ROOT}/data/drivers"
if [[ ! -d "${VAVAM_DIR}" ]]; then
    echo "Downloading vavam assets..."
    ./data/download_vavam_assets.sh --model vavam-b
    if [[ $? -ne 0 ]]; then
        echo "❌ Failed to download VAVAM models. Exiting."
        rm -rf "${VAVAM_DIR}"
        return 1
    fi
else
    echo "VAVAM models already present. Skipping download."
fi

# Install Wizard in development mode
echo "Installing Wizard in development mode..."
uv tool install --python 3.12 -e "${REPO_ROOT}/src/wizard"

# Ensure Hugging Face token is available (needed to download files)
if [[ -z "${HF_TOKEN}" ]]; then
    echo "❌ Hugging Face token (HF_TOKEN) not found in environment."
    echo "If you need to download files from Hugging Face, please set HF_TOKEN."
    return 1
fi

# Ensure that the hugging face cache is available
if [[ -z "${HF_HOME}" ]]; then
    echo "Note: Hugging Face cache directory (HF_HOME) not found in environment."
    FALLBACK_HF_HOME="$HOME/.cache/huggingface"
    echo "Falling back to default cache directory at $FALLBACK_HF_HOME"
    if [[ ! -d "$FALLBACK_HF_HOME" ]]; then
        echo "Creating Hugging Face cache directory at $FALLBACK_HF_HOME"
        mkdir -p "$FALLBACK_HF_HOME"
        if [[ $? -ne 0 ]]; then
            echo "❌ Failed to create Hugging Face cache directory at $FALLBACK_HF_HOME"
            return 1
        fi
    fi
fi

echo "Setup complete"
