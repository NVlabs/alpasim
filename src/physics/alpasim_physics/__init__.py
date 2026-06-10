# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0.common_pb2 import VersionId

VERSION_MESSAGE = VersionId(
    version_id="0.2.0",  # TODO: hook up to scm
    git_hash="<TODO>",  # TODO: hook up to scm
    grpc_api_version=API_VERSION_MESSAGE,
)

__all__ = ("__version__", "VERSION_MESSAGE")
