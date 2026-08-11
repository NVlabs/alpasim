# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""NRE server introspection for scene-affine dispatch.

Requires an NRE build that implements the ``GetLoadedScenes`` RPC
(added in NRE MR !3709).  Returns ``None`` on transient gRPC failures so
callers can distinguish "query failed" from "genuinely no cached scenes".
"""

from __future__ import annotations

import logging
import time

from alpasim_grpc.v0.common_pb2 import Empty
from alpasim_grpc.v0.sensorsim_pb2 import AvailableCamerasRequest, LoadedScenesReturn
from alpasim_grpc.v0.sensorsim_pb2_grpc import SensorsimServiceStub

import grpc
import grpc.aio

logger = logging.getLogger(__name__)

_INTROSPECTION_TIMEOUT_S = 15.0
_SCENE_PREFETCH_TIMEOUT_S = 300.0


class IntrospectionNotSupportedError(Exception):
    """Raised when the NRE server does not implement GetLoadedScenes."""

    pass


async def _query_loaded_scene_state(
    stub: SensorsimServiceStub,
    address: str,
    *,
    raise_on_unimplemented: bool = False,
) -> LoadedScenesReturn | None:
    """Query one connected NRE stub for its complete cache state."""
    try:
        return await stub.get_loaded_scenes(Empty(), timeout=_INTROSPECTION_TIMEOUT_S)
    except grpc.aio.AioRpcError as exc:
        if raise_on_unimplemented and exc.code() == grpc.StatusCode.UNIMPLEMENTED:
            raise IntrospectionNotSupportedError(
                f"NRE server at {address} does not support GetLoadedScenes "
                f"(UNIMPLEMENTED). Scene-affine dispatch requires an NRE image "
                f"with this RPC. Either upgrade the NRE image or set "
                f"scene_affine_dispatch.enabled=false."
            ) from exc
        logger.warning("get_loaded_scenes failed on %s: %s", address, exc.details())
        return None
    except Exception as exc:
        logger.warning("get_loaded_scenes unexpected error on %s: %s", address, exc)
        return None


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
        state = await _query_loaded_scene_state(
            stub,
            address,
            raise_on_unimplemented=raise_on_unimplemented,
        )
        if state is None:
            return None
        return {entry.scene_id: entry.loaded_instance_count for entry in state.scenes}
    finally:
        await channel.close()


async def prefetch_scene(address: str, scene_id: str) -> None:
    """Load one scene into NRE's reusable backend cache without rendering."""
    channel = grpc.aio.insecure_channel(address)
    started = time.monotonic()
    try:
        stub = SensorsimServiceStub(channel)
        before = await _query_loaded_scene_state(stub, address)
        if before is None:
            logger.warning(
                "Scene prefetch skipped because NRE cache state is unavailable: "
                "scene_id=%s renderer=%s",
                scene_id,
                address,
            )
            return

        before_count = next(
            (
                entry.loaded_instance_count
                for entry in before.scenes
                if entry.scene_id == scene_id
            ),
            0,
        )
        loaded_instances = sum(entry.loaded_instance_count for entry in before.scenes)
        reusable_instances = sum(
            entry.reusable_instance_count for entry in before.scenes
        )
        capacity = before.loaded_instance_capacity
        if before_count > 0:
            logger.info(
                "Scene prefetch skipped because scene is already loaded: "
                "scene_id=%s renderer=%s cache_instances=%d",
                scene_id,
                address,
                before_count,
            )
            return
        if capacity > 0 and loaded_instances >= capacity and reusable_instances == 0:
            logger.info(
                "Scene prefetch skipped because NRE cache is fully in use: "
                "scene_id=%s renderer=%s loaded_instances=%d "
                "reusable_instances=%d capacity=%d",
                scene_id,
                address,
                loaded_instances,
                reusable_instances,
                capacity,
            )
            return

        logger.info(
            "Scene prefetch NRE load starting: scene_id=%s renderer=%s "
            "cache_instances_before=%d loaded_instances=%d reusable_instances=%d "
            "capacity=%d",
            scene_id,
            address,
            before_count,
            loaded_instances,
            reusable_instances,
            capacity,
        )
        response = await stub.get_available_cameras(
            AvailableCamerasRequest(scene_id=scene_id),
            timeout=_SCENE_PREFETCH_TIMEOUT_S,
            wait_for_ready=True,
        )
        after = await _query_loaded_scene_state(stub, address)
        after_count = (
            None
            if after is None
            else next(
                (
                    entry.loaded_instance_count
                    for entry in after.scenes
                    if entry.scene_id == scene_id
                ),
                0,
            )
        )
        logger.info(
            "Scene prefetch NRE load complete: scene_id=%s renderer=%s "
            "duration_s=%.3f cache_instances_before=%d cache_instances_after=%s "
            "camera_count=%d",
            scene_id,
            address,
            time.monotonic() - started,
            before_count,
            after_count if after_count is not None else "unknown",
            len(response.available_cameras),
        )
    except Exception:
        logger.exception(
            "Scene prefetch NRE load failed: scene_id=%s renderer=%s duration_s=%.3f",
            scene_id,
            address,
            time.monotonic() - started,
        )
        raise
    finally:
        await channel.close()
