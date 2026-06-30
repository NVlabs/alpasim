# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from alpasim_grpc.v0.common_pb2 import VersionId

API_VERSION = (0, 54, 0)

API_VERSION_MESSAGE = VersionId.APIVersion(
    major=API_VERSION[0],
    minor=API_VERSION[1],
    patch=API_VERSION[2],
)
