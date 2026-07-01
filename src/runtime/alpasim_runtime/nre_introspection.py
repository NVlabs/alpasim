# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""NRE server introspection for scene-affine dispatch.

Requires an NRE build that implements the ``GetLoadedScenes`` RPC
(added in NRE MR !3709).  Returns ``None`` on transient gRPC failures so
callers can distinguish "query failed" from "genuinely no cached scenes".
"""

from __future__ import annotations

import logging

from alpasim_grpc.v0.common_pb2 import Empty
from alpasim_grpc.v0.sensorsim_pb2_grpc import SensorsimServiceStub

import grpc
import grpc.aio

logger = logging.getLogger(__name__)

_INTROSPECTION_TIMEOUT_S = 15.0


class IntrospectionNotSupportedError(Exception):
    """Raised when the NRE server does not implement GetLoadedScenes."""

    pass


async def get_loaded_scenes(
    address: str,
    *,
    raise_on_unimplemented: bool = False,
) -> dict[str, int] | None:
    """Query an NRE server for its currently loaded scene counts.

    Returns a ``{scene_id: loaded_instance_count}`` dict on success, or
    ``None`` if the RPC call fails for a transient reason (so callers can
    distinguish "no cached scenes" from "query failed").

    Args:
        address: gRPC address of the NRE server.
        raise_on_unimplemented: If True, raises ``IntrospectionNotSupportedError``
            when the server returns UNIMPLEMENTED instead of silently returning None.
            Use at startup to fail fast when scene-affine dispatch is enabled but
            the NRE image doesn't support introspection.
    """
    channel = grpc.aio.insecure_channel(address)
    try:
        stub = SensorsimServiceStub(channel)
        response = await stub.get_loaded_scenes(
            Empty(), timeout=_INTROSPECTION_TIMEOUT_S
        )
        return {
            entry.scene_id: entry.loaded_instance_count for entry in response.scenes
        }
    except grpc.aio.AioRpcError as e:
        if raise_on_unimplemented and e.code() == grpc.StatusCode.UNIMPLEMENTED:
            raise IntrospectionNotSupportedError(
                f"NRE server at {address} does not support GetLoadedScenes "
                f"(UNIMPLEMENTED). Scene-affine dispatch requires an NRE image "
                f"with this RPC. Either upgrade the NRE image or disable "
                f"scene_affine_dispatch."
            ) from e
        logger.warning("get_loaded_scenes failed on %s: %s", address, e.details())
        return None
    except Exception as e:
        logger.warning("get_loaded_scenes unexpected error on %s: %s", address, e)
        return None
    finally:
        await channel.close()
