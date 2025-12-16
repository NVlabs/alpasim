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
uv tool install -e "${REPO_ROOT}/src/wizard"

# Ensure Hugging Face token is available (needed to download files)
# Check if HF_TOKEN is set in the environment
if [[ -z "${HF_TOKEN}" ]]; then
    echo "❌ Hugging Face token (HF_TOKEN) not found in environment."
    echo "If you need to download files from Hugging Face, please set HF_TOKEN."
    return 1
fi

echo "Setup complete"
