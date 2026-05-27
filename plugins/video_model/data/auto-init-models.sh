#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
#
# Video-model plugin auto-init registrations. Sourced by data/auto-init.sh
# when this plugin is present. Adds an option to download the public scene
# pack used by the external video-model deploy.

# Public OmniDreams scenes (5 clipgt-* USDZ files, ~1.05 GB total) hosted on
# HuggingFace. Used by `deploy=external_video_model` when the wizard is
# pointed at `scenes.local_usdz_dir=$PWD/data/omni-dreams-scenes/scenes`.
model_labels+=( "OmniDreams Scenes (HuggingFace, nvidia-omni-dreams-lha/omni-dreams-scenes)" )
model_states+=( 0 )
model_commands+=(
  "uv run huggingface-cli download nvidia-omni-dreams-lha/omni-dreams-scenes --repo-type dataset --local-dir \"${SCRIPT_DIR}/omni-dreams-scenes\""
)
