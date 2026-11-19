# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Unit tests for SensorsimService batch (batch_render_rgb) response mapping."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from alpasim_grpc.v0.sensorsim_pb2 import (
    AggregatedRenderReturn,
    AvailableEgoMasksReturn,
    BatchRGBRenderReturnItem,
    EgoMaskId,
    RGBRenderRequest,
    RGBRenderReturn,
)
from alpasim_runtime.errors import InvalidSceneError
from alpasim_runtime.services.sensorsim_service import SensorsimService
from alpasim_runtime.services.service_base import SessionInfo
from alpasim_runtime.services.session_configs import RendererSessionConfig
from alpasim_runtime.types import Clock, RuntimeCamera


def _item(camera_name: str, image: bytes = b"img", success: bool = True, err: str = ""):
    return BatchRGBRenderReturnItem(
        camera_name=camera_name,
        result=RGBRenderReturn(image_bytes=image),
        success=success,
        error_message=err,
    )


def _triggers() -> dict[str, Clock.Trigger]:
    return {
        "cam_a": Clock.Trigger(range(0, 33_000), sequential_idx=0),
        "cam_b": Clock.Trigger(range(100_000, 133_000), sequential_idx=1),
    }


@pytest.mark.asyncio
async def test_sensorsim_service_uses_large_grpc_message_limits(monkeypatch):
    captured: dict[str, object] = {}
    fake_channel = object()

    def fake_insecure_channel(address, options=None):
        captured["address"] = address
        captured["options"] = options
        return fake_channel

    class FakeStub:
        def __init__(self, channel):
            captured["channel"] = channel

    monkeypatch.setattr(
        "alpasim_runtime.services.service_base.grpc.aio.insecure_channel",
        fake_insecure_channel,
    )
    monkeypatch.setattr(
        "alpasim_runtime.services.sensorsim_service.SensorsimServiceStub",
        FakeStub,
    )

    service = SensorsimService("localhost:50051", False, MagicMock())
    await service._open_connection()

    assert captured["address"] == "localhost:50051"
    assert captured["channel"] is fake_channel
    options = dict(captured["options"])
    assert options.keys() == {
        "grpc.max_receive_message_length",
        "grpc.max_send_message_length",
    }
    assert options["grpc.max_receive_message_length"] > 4 * 1024 * 1024
    assert (
        options["grpc.max_send_message_length"]
        == options["grpc.max_receive_message_length"]
    )


def test_batch_return_maps_images_by_camera_name():
    items = [_item("cam_a", b"a"), _item("cam_b", b"b")]

    images = SensorsimService._batch_return_to_images(items, _triggers())

    by_cam = {img.camera_logical_id: img for img in images}
    assert set(by_cam) == {"cam_a", "cam_b"}
    assert by_cam["cam_a"].image_bytes == b"a"
    # Timestamps come from the request-side trigger, not the response.
    assert by_cam["cam_a"].start_timestamp_us == 0
    assert by_cam["cam_a"].end_timestamp_us == 33_000
    assert by_cam["cam_b"].start_timestamp_us == 100_000


def test_batch_return_raises_on_failed_item():
    items = [
        _item("cam_a"),
        _item("cam_b", success=False, err="actor editing disabled"),
    ]

    with pytest.raises(RuntimeError, match=r"cam_b.*actor editing disabled"):
        SensorsimService._batch_return_to_images(items, _triggers())


def test_batch_return_raises_on_unknown_camera():
    items = [_item("cam_a"), _item("cam_UNEXPECTED")]

    with pytest.raises(RuntimeError, match=r"unknown camera 'cam_UNEXPECTED'"):
        SensorsimService._batch_return_to_images(items, _triggers())


def test_batch_return_raises_on_duplicate_camera():
    # cam_b would be missing the next loop, but the duplicate must fail first.
    items = [_item("cam_a"), _item("cam_a")]

    with pytest.raises(RuntimeError, match=r"duplicate camera 'cam_a'"):
        SensorsimService._batch_return_to_images(items, _triggers())


def test_batch_return_raises_on_missing_camera():
    # Requested cam_a and cam_b, but NRE only returned cam_a.
    items = [_item("cam_a")]

    with pytest.raises(RuntimeError, match=r"omitted requested camera.*cam_b"):
        SensorsimService._batch_return_to_images(items, _triggers())


@pytest.mark.asyncio
async def test_batch_render_skip_mode_returns_empty_images():
    """In skip mode batch_render returns placeholder frames like render()."""
    svc = SensorsimService("addr:0", skip=True, camera_catalog=MagicMock())
    cam = RuntimeCamera(
        logical_id="cam_a",
        render_resolution_hw=(2, 2),
        clock=Clock(interval_us=100_000, duration_us=33_000, start_us=0),
    )
    trigger = cam.clock.ith_trigger(0)

    images, driver_data = await svc.batch_render(
        [(cam, trigger)],
        ego_trajectory=MagicMock(),
        traffic_trajectories={},
        clip_id="clip",
        image_format=MagicMock(),
    )

    assert driver_data is None
    assert [img.camera_logical_id for img in images] == ["cam_a"]
    assert images[0].image_bytes == b""


def _mask_catalog(pairs):
    """Build the native renderer's bank/camera metadata response."""
    result = AvailableEgoMasksReturn()
    for bank, camera in pairs:
        result.ego_mask_metadata.add(
            ego_mask_id=EgoMaskId(
                rig_config_id=bank,
                camera_logical_id=camera,
            )
        )
    return result


@pytest.mark.asyncio
async def test_per_clip_mask_selection_is_session_local_and_camera_scoped():
    """Sharing one renderer/catalog does not leak a prior clip's bank or camera subset."""
    service = SensorsimService(
        "addr:0",
        False,
        SimpleNamespace(
            merge_local_and_sensorsim_cameras=AsyncMock(),
        ),
    )
    service.get_available_cameras = AsyncMock(return_value=[])
    catalog = _mask_catalog(
        [("first", "front"), ("first", "rear"), ("second", "front"), ("second", "rear")]
    )
    service.get_available_ego_masks = AsyncMock(return_value=catalog)
    for bank, selected in [("first", "front"), ("second", "rear")]:
        cfg = RendererSessionConfig(
            SimpleNamespace(scene_id=bank), [], MagicMock(), "jpeg", bank, [selected]
        )
        session = SessionInfo(bank, MagicMock(), cfg)
        service.session_info = session
        await service._initialize_session(session)
        for camera in ["front", "rear"]:
            mask = SensorsimService.determine_ego_mask_id(
                catalog, camera, bank, [selected]
            )
            assert (mask is not None) == (camera == selected)
            if mask is not None:
                assert mask.rig_config_id == bank
    # The global/default path retains its original camera discovery behavior.
    service.session_info = None
    assert (
        SensorsimService.determine_ego_mask_id(
            catalog, "rear", "first", None
        ).rig_config_id
        == "first"
    )
    assert (
        SensorsimService.determine_ego_mask_id(catalog, "missing", "first", None)
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pairs", [[("bank", "rear")], [("bank", "front"), ("bank", "front")]]
)
async def test_missing_or_ambiguous_override_mask_fails_at_session_start(pairs):
    """A configured texture must not silently fall back to an unmasked render."""
    service = SensorsimService(
        "addr:0",
        False,
        SimpleNamespace(
            merge_local_and_sensorsim_cameras=AsyncMock(),
        ),
    )
    service.get_available_cameras = AsyncMock(return_value=[])
    service.get_available_ego_masks = AsyncMock(return_value=_mask_catalog(pairs))
    cfg = RendererSessionConfig(
        SimpleNamespace(scene_id="clip"), [], MagicMock(), "jpeg", "bank", ["front"]
    )
    session = SessionInfo("clip", MagicMock(), cfg)
    service.session_info = session
    with pytest.raises(InvalidSceneError, match="requires exactly one mask"):
        await service._initialize_session(session)


@pytest.mark.asyncio
@pytest.mark.parametrize("pairs", [[], [("other", "front")]])
@pytest.mark.parametrize("selected_cameras", [None, [], ["front"]])
async def test_missing_mask_bank_fails_at_session_start(pairs, selected_cameras):
    service = SensorsimService(
        "addr:0",
        False,
        SimpleNamespace(merge_local_and_sensorsim_cameras=AsyncMock()),
    )
    service.get_available_cameras = AsyncMock(return_value=[])
    service.get_available_ego_masks = AsyncMock(return_value=_mask_catalog(pairs))
    cfg = RendererSessionConfig(
        SimpleNamespace(scene_id="clip"),
        [],
        MagicMock(),
        "jpeg",
        "bank",
        selected_cameras,
    )
    session = SessionInfo("clip", MagicMock(), cfg)
    with pytest.raises(
        InvalidSceneError, match="Ego texture bank 'bank'.*not available"
    ):
        await service._initialize_session(session)


@pytest.mark.asyncio
async def test_default_bank_with_only_front_mask_allows_other_cameras():
    service = SensorsimService(
        "addr:0",
        False,
        SimpleNamespace(merge_local_and_sensorsim_cameras=AsyncMock()),
    )
    service.get_available_cameras = AsyncMock(return_value=[])
    catalog = _mask_catalog([("bank", "front")])
    service.get_available_ego_masks = AsyncMock(return_value=catalog)
    cameras = [
        RuntimeCamera(
            logical_id=name,
            render_resolution_hw=(2, 2),
            clock=Clock(interval_us=100_000, duration_us=33_000, start_us=0),
        )
        for name in ["front", "rear"]
    ]
    cfg = RendererSessionConfig(
        SimpleNamespace(scene_id="clip"), cameras, MagicMock(), "jpeg", "bank"
    )
    await service._initialize_session(SessionInfo("clip", MagicMock(), cfg))

    assert SensorsimService.determine_ego_mask_id(catalog, "front", "bank", None)
    assert SensorsimService.determine_ego_mask_id(catalog, "rear", "bank", None) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("skip, bank", [(True, "bank"), (False, None)])
@pytest.mark.parametrize("selected_cameras", [None, [], ["front"]])
async def test_disabled_masks_do_not_validate_catalog(skip, bank, selected_cameras):
    service = SensorsimService(
        "addr:0",
        skip,
        SimpleNamespace(merge_local_and_sensorsim_cameras=AsyncMock()),
    )
    service.get_available_cameras = AsyncMock(return_value=[])
    service.get_available_ego_masks = AsyncMock(
        side_effect=AssertionError("disabled masks must not probe the catalog")
    )
    cfg = RendererSessionConfig(
        SimpleNamespace(scene_id="clip"),
        [],
        MagicMock(),
        "jpeg",
        bank,
        selected_cameras,
    )
    session = SessionInfo("clip", MagicMock(), cfg)
    service.session_info = session
    await service._initialize_session(session)
    service.get_available_ego_masks.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("response_count", [1, 2])
async def test_aggregated_render_response_mapping(monkeypatch, response_count):
    service = SensorsimService("addr:0", skip=False, camera_catalog=MagicMock())
    service.session_info = SessionInfo(
        uuid="session",
        broadcaster=SimpleNamespace(broadcast=AsyncMock()),
        session_config=RendererSessionConfig(
            MagicMock(), [], MagicMock(), "jpeg", None
        ),
    )
    service.stub = SimpleNamespace(render_aggregated=object())
    service.get_available_ego_masks = AsyncMock(return_value=AvailableEgoMasksReturn())
    service.determine_ego_mask_id = MagicMock(return_value=None)

    cameras = [
        RuntimeCamera(
            logical_id=name,
            render_resolution_hw=(2, 2),
            clock=Clock(interval_us=100_000, duration_us=33_000, start_us=0),
        )
        for name in ["cam_a", "cam_b"]
    ]
    triggers = list(zip(cameras, _triggers().values()))

    def construct_request(ego, traffic, camera, trigger, clip, image_format, ego_mask):
        request = RGBRenderRequest(
            frame_start_us=trigger.time_range_us.start,
            frame_end_us=trigger.time_range_us.stop,
        )
        request.camera_intrinsics.logical_id = camera.logical_id
        return request

    service.construct_rgb_render_request = construct_request
    response = AggregatedRenderReturn(
        rgb_returns=[
            RGBRenderReturn(image_bytes=value)
            for value in [b"a", b"b"][:response_count]
        ],
        driver_data=b"driver-payload",
    )
    service._sensorsim_rpc = AsyncMock(return_value=response)
    arguments = dict(
        camera_triggers=triggers,
        ego_trajectory=MagicMock(),
        traffic_trajectories={},
        clip_id="clip",
        image_format=MagicMock(),
    )
    if response_count != len(cameras):
        with pytest.raises(RuntimeError, match="camera triggers"):
            await service.aggregated_render(**arguments)
        return

    images, driver_data = await service.aggregated_render(**arguments)
    assert driver_data == b"driver-payload"
    assert [(image.camera_logical_id, image.image_bytes) for image in images] == [
        ("cam_a", b"a"),
        ("cam_b", b"b"),
    ]
    assert [(image.start_timestamp_us, image.end_timestamp_us) for image in images] == [
        (0, 33_000),
        (100_000, 133_000),
    ]


@pytest.mark.parametrize("mismatch", ["timestamp", "camera"])
def test_rgb_metadata_validation_rejects_mismatch(mismatch):
    camera = SimpleNamespace(logical_id="cam_a")
    trigger = _triggers()["cam_a"]
    request = RGBRenderRequest(frame_start_us=0, frame_end_us=33_000)
    request.camera_intrinsics.logical_id = "cam_a"
    if mismatch == "timestamp":
        request.frame_end_us += 1
    else:
        request.camera_intrinsics.logical_id = "cam_b"
    with pytest.raises(RuntimeError, match="does not match camera trigger"):
        SensorsimService._validate_rgb_metadata_matches_trigger(
            request, camera, trigger
        )
