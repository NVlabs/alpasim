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
VAVAM_DIR="${REPO_ROOT}/data/vavam-driver"
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

# Ensure Hugging Face cli is installed and logged in
# check for binary hf
if ! command -v hf &> /dev/null; then
    echo "Hugging Face CLI not found. Installing with pip install -U huggingface_hub? (y/n)"
    read -r response
    if [[ "$response" == "y" ]]; then
        pip install -U huggingface_hub
    else
        echo "❌ Hugging Face CLI is required. Exiting."
        return 1
    fi
fi

echo "Logging into Hugging Face CLI..."
hf auth login
if [[ $? -ne 0 ]]; then
    echo "❌ Failed to log into Hugging Face CLI. If you have a Hugging Face token,"
    echo "   you may not have sufficient privileges on that token. Exiting."
    return 1
fi

# Download the sample model to cache
hf download --repo-type=dataset \
    --local-dir=data/nre-artifacts/all-usdzs \
    nvidia/PhysicalAI-Autonomous-Vehicles-NuRec \
    sample_set/25.07_release/Batch0001/05bb8212-63e1-40a8-b4fc-3142c0e94646/05bb8212-63e1-40a8-b4fc-3142c0e94646.usdz
if [[ $? -ne 0 ]]; then
    echo "❌ Failed to download sample data from Hugging Face. If you have a Hugging Face token,"
    echo "   you may not have sufficient privileges on that token. Exiting."
    return 1
fi


# Install Wizard in development mode
echo "Installing Wizard in development mode..."
uv tool install -e "${REPO_ROOT}/src/wizard"

echo "Setup complete"
