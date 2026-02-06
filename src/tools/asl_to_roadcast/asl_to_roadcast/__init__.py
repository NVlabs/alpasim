# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
ASL to RoadCast (RCLog) conversion tool.

This package provides utilities for converting Alpasim Simulation Log (ASL) files
to the RoadCast log format (RCLog) used by internal AV tooling.

Note: This package requires authentication with internal PyPI servers to access
the maglev.av dependency. Use the included buildauth script to authenticate:

    ./buildauth static
    uv sync
    uv run asl-to-roadcast -i <input.asl> -o <output_dir>

Usage:
    from asl_to_roadcast.rclog_message_generator import AVMessageGenerator, RCLogLogWriter
"""

__all__ = ["rclog_message_generator", "message_converters", "actor", "config", "utils"]
