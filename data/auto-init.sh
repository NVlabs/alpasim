#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT="${SCRIPT_DIR}/.."

# ---------------------------------------------------------------------------
# Model downloads – toggle-based selection
# ---------------------------------------------------------------------------

# Each model: label, default state (1=on, 0=off), download command
model_labels=( "VaVAM-B"  "AR1 (HuggingFace, nvidia/Alpamayo-R1-10B)"  "A1.5 (HuggingFace, nvidia/Alpamayo-1.5-10B)" )
model_states=( 1          0                                             0 )
model_commands=(
  '"${SCRIPT_DIR}/download_vavam_assets.sh" --model vavam-b'
  'uv run huggingface-cli download nvidia/Alpamayo-R1-10B'
  'uv run huggingface-cli download nvidia/Alpamayo-1.5-10B && uv run huggingface-cli download nvidia/Cosmos-Reason2-8B'
)

# ---------------------------------------------------------------------------
# Plugin model registration – plugins can append to the arrays above
# ---------------------------------------------------------------------------
for _plugin_models in "${REPO_ROOT}"/plugins/*/data/auto-init-models.sh; do
  [[ -f "$_plugin_models" ]] && source "$_plugin_models"
done
unset _plugin_models

if [[ "${#model_labels[@]}" -ne "${#model_states[@]}" ]] || \
   [[ "${#model_labels[@]}" -ne "${#model_commands[@]}" ]]; then
  echo "ERROR: model_labels, model_states, and model_commands arrays have different lengths" >&2
  exit 1
fi

download_model() {
  eval "${model_commands[$1]}"
}

show_menu() {
  echo ""
  echo "Select models to download (toggle numbers, press Enter when done):"
  for i in "${!model_labels[@]}"; do
    if [[ "${model_states[$i]}" -eq 1 ]]; then
      mark="x"
    else
      mark=" "
    fi
    printf "  %d) [%s] %s\n" "$((i + 1))" "$mark" "${model_labels[$i]}"
  done
  echo ""
}

while true; do
  show_menu
  read -rp "Toggle [1-${#model_labels[@]}] or press Enter to confirm: " toggle
  if [[ -z "$toggle" ]]; then
    break
  fi
  if [[ "$toggle" =~ ^[0-9]+$ ]] && (( toggle >= 1 && toggle <= ${#model_labels[@]} )); then
    idx=$((toggle - 1))
    model_states[$idx]=$(( 1 - model_states[$idx] ))
  else
    echo "Invalid input '${toggle}', enter a number between 1 and ${#model_labels[@]}."
  fi
done

any_selected=0
for i in "${!model_labels[@]}"; do
  if [[ "${model_states[$i]}" -eq 1 ]]; then
    echo "Downloading ${model_labels[$i]} …"
    download_model "$i"
    any_selected=1
  fi
done

if [[ "$any_selected" -eq 0 ]]; then
  echo "No models selected – nothing to download."
fi
