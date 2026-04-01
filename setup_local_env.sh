#! /bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

# Ensure the script is sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "❌ This script must be sourced, not executed. Use:"
    echo "    source $0"
    exit 1
fi

# Get the repository root directory (based on script location)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


# Check for Rust toolchain (required for utils_rs maturin build)
if ! command -v cargo &> /dev/null; then
    echo "⚠️  Rust toolchain (cargo) not found. It is required for building utils_rs."
    if [[ -t 0 ]]; then
        read -p "Would you like to install it via rustup? [y/N] " -r
        if [[ "$REPLY" =~ ^[Yy]$ ]]; then
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
    else
        echo "❌ Rust toolchain is required. Install via: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
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

# Ensure Hugging Face token is available (needed to download files)
if [[ -z "${HF_TOKEN}" ]]; then
    echo "⚠️   Hugging Face token (HF_TOKEN) not found in environment."
    echo "If you need to download files from Hugging Face, please set HF_TOKEN."
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

# refresh utils_rs package (it doesn't auto-update because it's a compiled extension)
uv pip install --force-reinstall -e "${REPO_ROOT}/src/utils_rs"


# Install all core packages via extras, auto-detecting available plugins
cd "${REPO_ROOT}" || { echo "❌ Failed to change to repository root: ${REPO_ROOT}" >&2; return 1; }

EXTRAS=("--extra" "all")

# Map plugin directories to their pyproject.toml extra names
declare -A PLUGIN_EXTRAS=(
    ["plugins/internal"]="internal"
    ["plugins/transfuser_driver"]="transfuser"
)

echo "Detecting available plugins..."
for plugin_dir in "${!PLUGIN_EXTRAS[@]}"; do
    if [[ -d "${REPO_ROOT}/${plugin_dir}" && -f "${REPO_ROOT}/${plugin_dir}/pyproject.toml" ]]; then
        extra_name="${PLUGIN_EXTRAS[$plugin_dir]}"
        echo "  Found plugin: ${plugin_dir} (extra: ${extra_name})"
        EXTRAS+=("--extra" "${extra_name}")
    fi
done

echo "Installing packages with extras: ${EXTRAS[*]}"
uv sync "${EXTRAS[@]}"
if [[ $? -ne 0 ]]; then
    echo "⚠️  Failed to sync packages. You may need to check your environment."
fi


echo "Setup complete"
