# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from alpasim_grpc.v0 import (
    runtime_pb2,
    runtime_pb2_grpc,
    sensorsim_pb2,
    sensorsim_pb2_grpc,
)
from alpasim_runtime.address_pool import AddressPool
from alpasim_runtime.config import RendererKind
from alpasim_runtime.daemon.engine import DaemonEngine
from alpasim_runtime.daemon.servicer import RuntimeDaemonServicer

import grpc


class _CachingSensorsimServicer(sensorsim_pb2_grpc.SensorsimServiceServicer):
    """Minimal NRE double that exposes its cache transition through real RPCs."""

    def __init__(self) -> None:
        self.loaded_scene_ids: set[str] = set()
        self.events: list[str] = []

    async def get_loaded_scenes(self, request, context):
        del request, context
        self.events.append(f"cache:{sorted(self.loaded_scene_ids)!r}")
        return sensorsim_pb2.LoadedScenesReturn(
            scenes=[
                sensorsim_pb2.LoadedSceneEntry(
                    scene_id=scene_id,
                    loaded_instance_count=1,
                    reusable_instance_count=1,
                )
                for scene_id in sorted(self.loaded_scene_ids)
            ],
            loaded_instance_capacity=32,
        )

    async def get_available_cameras(self, request, context):
        del context
        self.events.append(f"load:{request.scene_id}")
        self.loaded_scene_ids.add(request.scene_id)
        return sensorsim_pb2.AvailableCamerasReturn(
            available_cameras=[
                sensorsim_pb2.AvailableCamerasReturn.AvailableCamera(
                    logical_id="camera_front_wide_120fov"
                )
            ]
        )


@pytest.mark.asyncio
async def test_runtime_prefetch_rpc_loads_scene_into_nre_cache(
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Exercise client -> RuntimeService -> engine -> NRE cache end to end."""
    scene_id = "clip-next-001"
    nre_servicer = _CachingSensorsimServicer()
    nre_server = grpc.aio.server()
    sensorsim_pb2_grpc.add_SensorsimServiceServicer_to_server(
        nre_servicer,
        nre_server,
    )
    nre_port = nre_server.add_insecure_port("127.0.0.1:0")
    await nre_server.start()

    engine = DaemonEngine(
        user_config="unused",
        network_config="unused",
        eval_config="unused",
        log_dir=str(tmp_path),
    )
    engine._scene_loader = SimpleNamespace(
        has_scene=lambda requested: requested == scene_id
    )
    engine._runtime_context = SimpleNamespace(
        config=SimpleNamespace(
            user=SimpleNamespace(
                renderer=SimpleNamespace(kind=RendererKind.sensorsim),
            )
        ),
        pools={
            "renderer": AddressPool(
                [f"127.0.0.1:{nre_port}"],
                n_concurrent=4,
                skip=False,
            )
        },
    )
    engine._started = True

    runtime_server = grpc.aio.server()
    runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeDaemonServicer(engine),
        runtime_server,
    )
    runtime_port = runtime_server.add_insecure_port("127.0.0.1:0")
    await runtime_server.start()

    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{runtime_port}") as channel:
            stub = runtime_pb2_grpc.RuntimeServiceStub(channel)
            with caplog.at_level(logging.INFO):
                await stub.prefetch_scene(
                    runtime_pb2.ScenePrefetchRequest(scene_id=scene_id),
                    timeout=5.0,
                )
    finally:
        await runtime_server.stop(grace=None)
        await nre_server.stop(grace=None)

    assert nre_servicer.events == [
        "cache:[]",
        f"load:{scene_id}",
        f"cache:['{scene_id}']",
    ]
    assert nre_servicer.loaded_scene_ids == {scene_id}
    assert "Scene prefetch request accepted" in caplog.text
    assert "Scene prefetch NRE load starting" in caplog.text
    assert "Scene prefetch NRE load complete" in caplog.text
    assert "cache_instances_before=0" in caplog.text
    assert "cache_instances_after=1" in caplog.text
