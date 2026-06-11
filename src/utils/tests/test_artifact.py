# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

from alpasim_utils.artifact import Artifact


def test_artifact_no_map():
    usdz_file = "tests/data/no_map_artifact/artifact_no_map.usdz"
    artifact = Artifact(source=usdz_file)

    # expect that the map is None (no exceptions)
    assert artifact.map is None


def test_xodr_artifact():
    usdz_file = "tests/data/xodr_artifact/026d6a39-bd8f-4175-bc61-fe50ed0403a3.usdz"
    artifact = Artifact(source=usdz_file)

    # expect that the map is not None (no exceptions)
    assert artifact.map is not None
    assert (
        artifact.map.map_id
        == "alpasim_usdz:clipgt-026d6a39-bd8f-4175-bc61-fe50ed0403a3"
    )
