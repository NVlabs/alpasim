# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Benchmark suite for the controller MPC performance.

Usage:
    # From src/controller directory:
    uv run python -m benchmark run --quick --output results/baseline.json
    uv run python -m benchmark compare results/baseline.json results/optimized.json -o report/

See benchmark/__main__.py for full CLI documentation.
"""
