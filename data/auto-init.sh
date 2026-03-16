#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# ---------------------------------------------------------------------------
# Model downloads – toggle-based selection
# ---------------------------------------------------------------------------

# Each model: label, default state (1=on, 0=off), download function body
model_labels=( "VaVAM-B"  "AR1 (HuggingFace, nvidia/Alpamayo-R1-10B)" )
model_states=( 1          0 )

if [[ "${#model_labels[@]}" -ne "${#model_states[@]}" ]]; then
  echo "ERROR: model_labels and model_states arrays have different lengths" >&2
  exit 1
fi

download_model() {
  case "$1" in
    0) "${SCRIPT_DIR}/download_vavam_assets.sh" --model vavam-b ;;
    1) huggingface-cli download nvidia/Alpamayo-R1-10B ;;
  esac
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
