# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""External video-model renderer plugin for Alpasim.

Registers ``video_model`` under the ``alpasim.services`` entry-point group so
core's worker can construct the renderer service and dispatch rendering to this
plugin when ``renderer_type: video_model``.
"""
