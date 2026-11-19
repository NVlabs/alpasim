# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import asyncio
import io
import json
from pathlib import Path

import pytest
import yaml
from alpasim_grpc.v0 import sensorsim_pb2, sensorsim_pb2_grpc, video_model_pb2_grpc
from alpasim_grpc.v0.common_pb2 import AvailableScenesReturn, Empty, VersionId
from alpasim_grpc.v0.video_model_pb2 import (
    CameraOutput,
    Image,
    ImageFormat,
    SessionCloseRequest,
    SessionId,
    SessionRequest,
    VideoChunkRequest,
    VideoChunkReturn,
)
from alpasim_runtime.simulate.__main__ import create_arg_parser, run_simulation
from alpasim_utils.logs import LogWriter
from PIL import Image as PILImage

import grpc
import grpc.aio

_TESTS_DIR = Path(__file__).parent
_MOCK_DATA_DIR = _TESTS_DIR / "data" / "mock"
_VIDEO_MODEL_DATA_DIR = _TESTS_DIR / "data" / "mock_video_model"
_TINY_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00\x01\x00\x01\x00\x00\xff\xd9"
)
_SENSORSIM_SCENE_ID = "clipgt-1ea7dc88-88ed-4c91-81fe-b6eb489cfa71"
_MOCK_EGO_MASK_RIG_CONFIG_ID = "hyperion_8_1"
_VIDEO_MODEL_SCENE_ID = "clipgt-0b10bce8-61f1-4350-8577-cf3c9493ffc3"
_VIDEO_MODEL_UUID = "162a328d-c9d1-4d39-bcc9-9ddffc2d142e"
_MOCK_CAMERA_IDS = (
    "camera_front_wide_120fov",
    "camera_front_tele_30fov",
)


def _write_run_metadata(log_dir: Path, run_name: str) -> None:
    (log_dir / "run_metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "run_uuid": f"{run_name}-uuid",
                "run_name": run_name,
            }
        ),
        encoding="utf-8",
    )


def _make_available_cameras() -> sensorsim_pb2.AvailableCamerasReturn:
    response = sensorsim_pb2.AvailableCamerasReturn()
    for logical_id in _MOCK_CAMERA_IDS:
        camera = response.available_cameras.add(logical_id=logical_id)
        camera.intrinsics.logical_id = logical_id
        camera.intrinsics.resolution_h = 320
        camera.intrinsics.resolution_w = 512
        camera.intrinsics.shutter_type = sensorsim_pb2.ShutterType.GLOBAL
        camera.intrinsics.opencv_pinhole_param.focal_length_x = 800.0
        camera.intrinsics.opencv_pinhole_param.focal_length_y = 800.0
        camera.intrinsics.opencv_pinhole_param.principal_point_x = 256.0
        camera.intrinsics.opencv_pinhole_param.principal_point_y = 160.0
        camera.rig_to_camera.quat.w = 1.0
    return response


class _FakeSensorsimServicer(sensorsim_pb2_grpc.SensorsimServiceServicer):
    def __init__(self) -> None:
        self.available_cameras_requests: list[sensorsim_pb2.AvailableCamerasRequest] = (
            []
        )
        self.render_requests: list[sensorsim_pb2.RGBRenderRequest] = []
        self.batch_render_requests: list[sensorsim_pb2.BatchRGBRenderRequest] = []
        self.aggregated_render_requests: list[sensorsim_pb2.AggregatedRenderRequest] = (
            []
        )

    async def get_version(self, request: Empty, context):
        del request, context
        return VersionId(version_id="fake-sensorsim", git_hash="test")

    async def get_available_scenes(self, request: Empty, context):
        del request, context
        return AvailableScenesReturn(scene_ids=[_SENSORSIM_SCENE_ID])

    async def get_available_cameras(
        self,
        request: sensorsim_pb2.AvailableCamerasRequest,
        context,
    ):
        del context
        self.available_cameras_requests.append(request)
        return _make_available_cameras()

    async def get_available_ego_masks(self, request: Empty, context):
        del request, context
        response = sensorsim_pb2.AvailableEgoMasksReturn()
        for logical_id in _MOCK_CAMERA_IDS:
            response.ego_mask_metadata.add(
                ego_mask_id=sensorsim_pb2.EgoMaskId(
                    rig_config_id=_MOCK_EGO_MASK_RIG_CONFIG_ID,
                    camera_logical_id=logical_id,
                )
            )
        return response

    async def render_rgb(self, request: sensorsim_pb2.RGBRenderRequest, context):
        del context
        self.render_requests.append(request)
        return sensorsim_pb2.RGBRenderReturn(image_bytes=_TINY_JPEG)

    async def batch_render_rgb(
        self,
        request: sensorsim_pb2.BatchRGBRenderRequest,
        context,
    ):
        del context
        self.batch_render_requests.append(request)
        response = sensorsim_pb2.BatchRGBRenderReturn()
        for item in request.items:
            ret_item = response.items.add()
            ret_item.camera_name = item.camera_name
            ret_item.result.image_bytes = _TINY_JPEG
            ret_item.success = True
        return response

    async def render_aggregated(
        self,
        request: sensorsim_pb2.AggregatedRenderRequest,
        context,
    ):
        del context
        self.aggregated_render_requests.append(request)
        response = sensorsim_pb2.AggregatedRenderReturn()
        for _ in request.rgb_requests:
            response.rgb_returns.add().image_bytes = _TINY_JPEG
        return response


async def _start_fake_sensorsim_server(
    servicer: _FakeSensorsimServicer,
) -> tuple[grpc.aio.Server, str]:
    server = grpc.aio.server()
    sensorsim_pb2_grpc.add_SensorsimServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    return server, f"127.0.0.1:{port}"


def _write_sensorsim_mock_configs(
    tmp_path: Path,
    sensorsim_address: str,
    *,
    telemetry_worker_port: int,
    batch_render: bool = False,
) -> dict:
    base_user_config = yaml.safe_load(
        (_MOCK_DATA_DIR / "user-config.yaml").read_text(encoding="utf-8")
    )
    base_user_config["data_source_provider"]["usdz"]["data_dir"] = str(_MOCK_DATA_DIR)
    base_user_config["endpoints"]["renderer"]["skip"] = False
    base_user_config["endpoints"]["renderer"]["n_concurrent_rollouts"] = 1
    base_user_config["prometheus"] = {
        "worker_ports": [telemetry_worker_port],
        "url": "http://127.0.0.1:9090",
    }
    base_user_config["simulation_config"]["n_sim_steps"] = 1
    if batch_render:
        base_user_config["simulation_config"]["render_bundling"] = "BATCH_RENDER_RGB"

    network_config = {
        service: {"endpoints": []}
        for service in (
            "controller",
            "driver",
            "physics",
            "trafficsim",
        )
    }
    network_config["renderer"] = {
        "endpoints": [{"address": sensorsim_address, "managed": False}]
    }

    user_config_path = tmp_path / "sensorsim-user-config.yaml"
    network_config_path = tmp_path / "sensorsim-network-config.yaml"
    eval_config_path = tmp_path / "sensorsim-eval-config.yaml"
    user_config_path.write_text(yaml.safe_dump(base_user_config), encoding="utf-8")
    network_config_path.write_text(yaml.safe_dump(network_config), encoding="utf-8")
    eval_config_path.write_text(
        (_MOCK_DATA_DIR / "eval-config.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    return {
        "user_config": str(user_config_path),
        "network_config": str(network_config_path),
        "eval_config": str(eval_config_path),
    }


@pytest.mark.asyncio
async def test_sensorsim_mocks_with_fake_server(tmp_path: Path):
    servicer = _FakeSensorsimServicer()
    server, address = await _start_fake_sensorsim_server(servicer)
    try:
        configs = _write_sensorsim_mock_configs(
            tmp_path,
            address,
            telemetry_worker_port=0,
        )

        _write_run_metadata(tmp_path, "test_sensorsim_mocks")

        parser = create_arg_parser()
        parsed_args = parser.parse_args(
            [
                f"--user-config={configs['user_config']}",
                f"--network-config={configs['network_config']}",
                f"--eval-config={configs['eval_config']}",
                f"--log-dir={tmp_path}",
            ]
        )

        success = await asyncio.wait_for(run_simulation(parsed_args), timeout=120)
    finally:
        await server.stop(grace=0.5)

    assert success
    assert servicer.available_cameras_requests
    assert servicer.available_cameras_requests[0].scene_id == _SENSORSIM_SCENE_ID
    assert servicer.render_requests
    assert {
        request.camera_intrinsics.logical_id for request in servicer.render_requests
    } == set(_MOCK_CAMERA_IDS)
    for request in servicer.render_requests:
        assert request.insert_ego_mask
        assert request.ego_mask_id.rig_config_id == _MOCK_EGO_MASK_RIG_CONFIG_ID
        assert (
            request.ego_mask_id.camera_logical_id
            == request.camera_intrinsics.logical_id
        )


@pytest.mark.asyncio
async def test_sensorsim_mocks_batch_render(tmp_path: Path):
    """render_bundling=BATCH_RENDER_RGB drives NRE batch_render_rgb."""
    servicer = _FakeSensorsimServicer()
    server, address = await _start_fake_sensorsim_server(servicer)
    try:
        configs = _write_sensorsim_mock_configs(
            tmp_path,
            address,
            telemetry_worker_port=0,
            batch_render=True,
        )

        _write_run_metadata(tmp_path, "test_sensorsim_batch")

        parser = create_arg_parser()
        parsed_args = parser.parse_args(
            [
                f"--user-config={configs['user_config']}",
                f"--network-config={configs['network_config']}",
                f"--eval-config={configs['eval_config']}",
                f"--log-dir={tmp_path}",
            ]
        )

        success = await asyncio.wait_for(run_simulation(parsed_args), timeout=120)
    finally:
        await server.stop(grace=0.5)

    assert success
    # Bundled path used: batch_render_rgb received, per-camera render_rgb not.
    assert servicer.batch_render_requests
    assert not servicer.render_requests
    rendered_cameras = {
        item.request.camera_intrinsics.logical_id
        for request in servicer.batch_render_requests
        for item in request.items
    }
    assert rendered_cameras == set(_MOCK_CAMERA_IDS)
    for batch in servicer.batch_render_requests:
        for item in batch.items:
            assert item.request.insert_ego_mask
            assert (
                item.request.ego_mask_id.rig_config_id == _MOCK_EGO_MASK_RIG_CONFIG_ID
            )
            assert (
                item.request.ego_mask_id.camera_logical_id
                == item.request.camera_intrinsics.logical_id
            )


class _FakeWorldModelServicer(video_model_pb2_grpc.WorldModelServiceServicer):
    def __init__(self) -> None:
        self.start_session_request: SessionRequest | None = None
        self.render_requests: list[VideoChunkRequest] = []
        self.close_session_request: SessionCloseRequest | None = None
        self._camera_ids: list[str] = []
        self._return_hdmap_frames = False

    async def get_version(self, request: Empty, context):
        del request, context
        return VersionId(version_id="fake-video-model", git_hash="test")

    async def start_session(self, request: SessionRequest, context):
        del context
        self.start_session_request = request
        self._camera_ids = [camera.logical_id for camera in request.camera_specs]
        self._return_hdmap_frames = request.debug_options.return_hdmap_frames
        return SessionId(session_id="fake-video-model-session")

    async def render_video_chunk(self, request: VideoChunkRequest, context):
        del context
        self.render_requests.append(request)
        frame_count = len(request.rig_trajectory.poses)
        camera_outputs = []
        for camera_id in self._camera_ids:
            output = CameraOutput(
                camera_logical_id=camera_id,
                rgb_frames=[
                    Image(data=_TINY_JPEG, format=ImageFormat.JPEG)
                    for _ in range(frame_count)
                ],
            )
            if self._return_hdmap_frames:
                output.hdmap_condition_frames.extend(
                    Image(data=_TINY_JPEG, format=ImageFormat.JPEG)
                    for _ in range(frame_count)
                )
            camera_outputs.append(output)
        return VideoChunkReturn(camera_outputs=camera_outputs)

    async def close_session(self, request: SessionCloseRequest, context):
        del context
        self.close_session_request = request
        return Empty()


async def _start_fake_video_model_server(
    servicer: _FakeWorldModelServicer,
) -> tuple[grpc.aio.Server, str]:
    server = grpc.aio.server()
    video_model_pb2_grpc.add_WorldModelServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    return server, f"127.0.0.1:{port}"


def _write_video_model_mock_configs(
    tmp_path: Path,
    video_model_address: str,
    *,
    telemetry_worker_port: int,
) -> dict:
    base_user_config = yaml.safe_load(
        (_MOCK_DATA_DIR / "user-config.yaml").read_text(encoding="utf-8")
    )
    base_user_config["data_source_provider"]["usdz"]["data_dir"] = str(
        _VIDEO_MODEL_DATA_DIR
    )
    base_user_config["scenes"] = [
        {
            "source": {
                "clip_id": _VIDEO_MODEL_SCENE_ID,
                "usdz_uuid": _VIDEO_MODEL_UUID,
            },
            "policy_start_us": "AUTO",
        }
    ]
    base_user_config.pop("extra_cameras", None)
    base_user_config["endpoints"]["renderer"]["skip"] = False
    base_user_config["endpoints"]["renderer"]["n_concurrent_rollouts"] = 1
    base_user_config["prometheus"] = {
        "worker_ports": [telemetry_worker_port],
        "url": "http://127.0.0.1:9090",
    }
    base_user_config["renderer"] = {
        "kind": "video_model",
        "video_model_config": {
            "fps": 10,
            "first_chunk_frames": 1,
            "chunk_frames": 1,
            "return_hdmap_frames": True,
            "forward_hdmap_to_driver": False,
        },
    }
    base_user_config["simulation_config"]["control_timestep_us"] = 100_000
    base_user_config["simulation_config"]["force_gt_duration_us"] = 200_000
    base_user_config["simulation_config"]["n_sim_steps"] = 1

    network_config = {
        service: {"endpoints": []}
        for service in (
            "controller",
            "driver",
            "physics",
            "trafficsim",
        )
    }
    network_config["renderer"] = {
        "endpoints": [{"address": video_model_address, "managed": False}]
    }

    user_config_path = tmp_path / "video-model-user-config.yaml"
    network_config_path = tmp_path / "video-model-network-config.yaml"
    eval_config_path = tmp_path / "video-model-eval-config.yaml"
    user_config_path.write_text(yaml.safe_dump(base_user_config), encoding="utf-8")
    network_config_path.write_text(yaml.safe_dump(network_config), encoding="utf-8")
    eval_config_path.write_text(
        (_MOCK_DATA_DIR / "eval-config.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    return {
        "user_config": str(user_config_path),
        "network_config": str(network_config_path),
        "eval_config": str(eval_config_path),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_close", [False, True])
async def test_video_model_mocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_close: bool
):
    image = io.BytesIO()
    PILImage.new("RGB", (512, 320), color=(128, 128, 128)).save(image, format="JPEG")
    monkeypatch.setattr(__name__ + "._TINY_JPEG", image.getvalue())

    original_close = LogWriter.__aexit__

    async def close_log(self, *args):
        assert not list(tmp_path.glob("rollouts/**/metrics.parquet"))
        assert not list(tmp_path.glob("rollouts/**/_complete"))
        await original_close(self, *args)
        if fail_close:
            raise OSError("log close failed")

    monkeypatch.setattr(LogWriter, "__aexit__", close_log)

    servicer = _FakeWorldModelServicer()
    server, address = await _start_fake_video_model_server(servicer)
    try:
        configs = _write_video_model_mock_configs(
            tmp_path,
            address,
            telemetry_worker_port=0,
        )

        user_path = Path(configs["user_config"])
        user = yaml.safe_load(user_path.read_text())
        user["prometheus"]["url"] = None
        user["max_rollout_retries"] = 0
        user_path.write_text(yaml.safe_dump(user))
        eval_path = Path(configs["eval_config"])
        config = yaml.safe_load(eval_path.read_text())
        config.update(enabled=True, allow_aggregation_with_failed_rollouts=True)
        config["aggregation_modifiers"]["remove_preengagement_timesteps"] = False
        config["scene_score"] = {"enabled": True}
        config["scorers"]["open_loop_collision"] = {"horizon_s": 3.0}
        eval_path.write_text(yaml.safe_dump(config))

        _write_run_metadata(tmp_path, "test_video_model_mocks")

        parser = create_arg_parser()
        parsed_args = parser.parse_args(
            [
                f"--user-config={configs['user_config']}",
                f"--network-config={configs['network_config']}",
                f"--eval-config={configs['eval_config']}",
                f"--log-dir={tmp_path}",
            ]
        )

        success = await asyncio.wait_for(run_simulation(parsed_args), timeout=120)
    finally:
        await server.stop(grace=0.5)

    assert success
    assert servicer.start_session_request is not None
    assert len(servicer.start_session_request.camera_specs) == 2
    assert len(servicer.start_session_request.initial_frames) == 2
    assert servicer.start_session_request.debug_options.return_hdmap_frames is True
    assert len(servicer.start_session_request.static_world_map.hdmap_parquets) > 0
    assert servicer.render_requests
    assert servicer.close_session_request is not None
    assert servicer.close_session_request.session_id == "fake-video-model-session"
    [asl] = tmp_path.glob("rollouts/**/rollout.asl")
    assert asl.stat().st_size > 0
    assert (asl.parent / "metrics.parquet").exists() == (not fail_close)
    assert (asl.parent / "_complete").exists() == (not fail_close)
    summary = json.loads((tmp_path / "aggregate/results-summary.json").read_text())
    [row] = summary["rollouts"]
    assert row["rollout_id"] == asl.parent.name
    assert row["status"] == ("fail" if fail_close else "pass")
    if not fail_close:
        assert row["score"] > 0
    else:
        assert row["score"] == 0
        assert "log close failed" in row["failure_reason"]
