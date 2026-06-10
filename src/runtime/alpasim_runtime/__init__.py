# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

__version__ = (0, 3, 0)

import os
import pathlib

from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0.common_pb2 import VersionId

source_directory = pathlib.Path(__file__).parent.resolve()
git_hash = os.environ.get("GIT_HASH", "Unknown")
if os.environ.get("GIT_DIRTY", "false") == "true":
    git_hash = git_hash + "+dirty"

VERSION_MESSAGE = VersionId(
    version_id=".".join(str(v) for v in __version__),
    git_hash=git_hash,
    grpc_api_version=API_VERSION_MESSAGE,
)

__all__ = ("__version__", "VERSION_MESSAGE")
