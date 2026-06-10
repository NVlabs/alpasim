# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

from alpasim_controller.server import construct_version


def test_construct_version():
    """Ensure version is at least filled."""
    version = construct_version()
    assert version.version_id is not None
    assert version.grpc_api_version is not None
    assert version.git_hash is not None
