# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

from pathlib import Path

from map_utils.plot_map import main


def test_sanity():
    # Resolve test data relative to the tools package root to work in or out of CI
    tools_root = Path(__file__).resolve().parents[1]
    artifact_name = tools_root / "tests/data/last.usdz"
    preview_route = True
    no_block = True
    # run, no assertion except for no exceptions
    main(str(artifact_name), preview_route, no_block)
