# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for the built-in video-model renderer.

Covers:
- Timing, chunking, and runtime wiring for the core video-model renderer.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from alpasim_grpc.v0.common_pb2 import Pose, PoseAtTime, Quat, Trajectory, Vec3
from alpasim_grpc.v0.video_model_pb2 import DynamicWorldState
from alpasim_runtime.config import SimulationConfig, VideoModelConfig
from alpasim_runtime.events.base import EventQueue
from alpasim_runtime.events.video_model.prefetch import (
    VideoModelPrefetchEvent,
    make_initial_video_model_render_event,
)
from alpasim_runtime.services.service_base import SessionInfo
from alpasim_runtime.services.session_configs import RendererSessionConfig
from alpasim_runtime.services.video_model_service import VideoModelService
from alpasim_runtime.types import Clock, RuntimeCamera


@pytest.mark.asyncio
async def test_video_model_service_uses_large_grpc_message_limits(monkeypatch) -> None:
    from alpasim_runtime.services.video_model_service import (
        MAX_GRPC_MESSAGE_BYTES,
        VideoModelService,
    )

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
        "alpasim_runtime.services.video_model_service.grpc.aio.insecure_channel",
        fake_insecure_channel,
    )
    monkeypatch.setattr(
        "alpasim_runtime.services.video_model_service.video_model_pb2_grpc.WorldModelServiceStub",
        FakeStub,
    )

    service = VideoModelService("localhost:50056", VideoModelConfig())
    await service._open_connection()

    assert captured["address"] == "localhost:50056"
    assert captured["channel"] is fake_channel
    assert captured["options"] == [
        ("grpc.max_receive_message_length", MAX_GRPC_MESSAGE_BYTES),
        ("grpc.max_send_message_length", MAX_GRPC_MESSAGE_BYTES),
    ]


def test_video_model_runtime_config_rejects_unknown_field() -> None:
    with pytest.raises(TypeError, match="definitely_not_a_field"):
        VideoModelService.from_config(
            raw_config={"definitely_not_a_field": 42},
            address="localhost:0",
            skip=True,
        )


def test_deploy_yaml_files_present() -> None:
    """The deploy + chunking YAMLs ship with the wizard's core configs.

    Note: cameras intentionally have no renderer-level config group -- the
    driver config (driver=vavam / driver=alpamayo1_5 / ...) owns the
    camera rig + rectification calibration; injecting a separate +cameras=
    override on top of that would mismatch the renderer output and the
    driver's rectifier.
    """
    from pathlib import Path

    root = Path(__file__).parents[3] / "wizard" / "configs"
    assert (root / "deploy" / "external_video_model.yaml").exists()

    assert (root / "chunking" / "8frame.yaml").exists()
    assert (root / "chunking" / "12frame.yaml").exists()
    assert (root / "chunking" / "16frame.yaml").exists()


def _load(rel: str):
    """Load a core wizard config by path relative to ``src/wizard/configs``."""
    from pathlib import Path

    from omegaconf import OmegaConf

    root = Path(__file__).parents[3] / "wizard" / "configs"
    return OmegaConf.load(root / rel)


def _video_model_service(
    *,
    fps: int = 30,
    first_chunk_frames: int = 5,
    chunk_frames: int = 8,
) -> VideoModelService:
    return VideoModelService(
        address="localhost:0",
        skip=True,
        config=VideoModelConfig(
            fps=fps,
            first_chunk_frames=first_chunk_frames,
            chunk_frames=chunk_frames,
        ),
    )


def _simulation_config(**overrides) -> SimulationConfig:
    cfg = SimulationConfig(n_sim_steps=1, n_rollouts=1)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _trajectory(timestamps_us: list[int]) -> Trajectory:
    return Trajectory(
        poses=[
            PoseAtTime(
                pose=Pose(
                    vec=Vec3(x=0.0, y=0.0, z=0.0),
                    quat=Quat(w=1.0, x=0.0, y=0.0, z=0.0),
                ),
                timestamp_us=t_us,
            )
            for t_us in timestamps_us
        ]
    )


def test_video_model_rejects_control_timestep_that_differs_from_chunk() -> None:
    with pytest.raises(ValueError, match="control_timestep_us"):
        _video_model_service().validate_timing_alignment(
            _simulation_config(
                control_timestep_us=100_000,
                force_gt_duration_us=2_033_313,
            )
        )


@pytest.mark.asyncio
async def test_skipped_video_model_render_chunk_returns_placeholder_frames() -> None:
    service = _video_model_service(fps=10)
    runtime_cameras = [
        RuntimeCamera(
            logical_id="camera_front",
            render_resolution_hw=(160, 256),
            clock=Clock(interval_us=100_000, duration_us=10_000),
        ),
        RuntimeCamera(
            logical_id="camera_left",
            render_resolution_hw=(160, 256),
            clock=Clock(interval_us=100_000, duration_us=10_000),
        ),
    ]
    await service._initialize_session(
        SessionInfo(
            uuid="session",
            broadcaster=SimpleNamespace(),
            session_config=RendererSessionConfig(
                data_source=SimpleNamespace(scene_id="scene"),
                runtime_cameras=runtime_cameras,
                gt_ego_trajectory=SimpleNamespace(),
                image_format="jpeg",
                ego_mask_rig_config_id=None,
            ),
        )
    )

    chunk = await service.render_chunk(
        trajectory_local_to_rig=_trajectory([100_000, 200_000]),
        dynamic_state=DynamicWorldState(),
    )

    assert set(chunk.rgb_frames_per_camera) == {"camera_front", "camera_left"}
    assert [
        f.start_timestamp_us for f in chunk.rgb_frames_per_camera["camera_front"]
    ] == [100_000, 200_000]
    assert all(
        f.image_bytes == b""
        for frames in chunk.rgb_frames_per_camera.values()
        for f in frames
    )


def test_video_model_rejects_force_gt_shorter_than_first_plus_regular() -> None:
    with pytest.raises(ValueError, match="first chunk plus"):
        _video_model_service().validate_timing_alignment(
            _simulation_config(
                control_timestep_us=266_664,
                force_gt_duration_us=166_665,
            )
        )


def test_video_model_rejects_force_gt_not_first_plus_regular_grid() -> None:
    with pytest.raises(ValueError, match="first_chunk_frames"):
        _video_model_service().validate_timing_alignment(
            _simulation_config(
                control_timestep_us=266_664,
                force_gt_duration_us=2_000_000,
            )
        )


@pytest.mark.parametrize(
    ("first_chunk_frames", "chunk_frames", "control_timestep_us", "force_gt_us"),
    [
        (5, 8, 266_664, 2_033_313),
        (9, 12, 399_996, 1_899_981),
        (13, 16, 533_328, 2_033_313),
    ],
)
def test_video_model_accepts_updated_chunking_configs(
    first_chunk_frames: int,
    chunk_frames: int,
    control_timestep_us: int,
    force_gt_us: int,
) -> None:
    service = _video_model_service(
        first_chunk_frames=first_chunk_frames,
        chunk_frames=chunk_frames,
    )

    service.validate_timing_alignment(
        _simulation_config(
            control_timestep_us=control_timestep_us,
            force_gt_duration_us=force_gt_us,
        )
    )

    assert service.required_policy_start_timestmap_us(150_000) == (
        150_000 + first_chunk_frames * 33_333
    )


def test_chunking_files_keep_first_chunk_and_control_timestep_in_sync() -> None:
    """first_chunk_frames + control_timestep_us are tied to chunk_frames.

    Server contract: first = (chunk // 4 - 1) * 4 + 1 (VAE temporal compression).
    Control timestep at 30 FPS: chunk * 33_333 us.
    force_gt_duration_us must be first_chunk_frames * frame_interval_us plus
    an integer number of regular control chunks.
    """
    frame_interval_us = 33_333
    for chunk_frames, expected_first, expected_control, expected_force_gt in [
        (8, 5, 266_664, 2_033_313),
        (12, 9, 399_996, 1_899_981),
        (16, 13, 533_328, 2_033_313),
    ]:
        cfg = _load(f"chunking/{chunk_frames}frame.yaml")
        rc = cfg.runtime.renderer.video_model_config
        sc = cfg.runtime.simulation_config
        assert rc.chunk_frames == chunk_frames
        assert rc.first_chunk_frames == expected_first
        assert sc.control_timestep_us == expected_control
        assert sc.force_gt_duration_us == expected_force_gt
        assert (
            sc.force_gt_duration_us - rc.first_chunk_frames * frame_interval_us
        ) % sc.control_timestep_us == 0


def test_external_video_model_keeps_zero_decision_delay() -> None:
    cfg = _load("deploy/external_video_model.yaml")

    assert cfg.runtime.simulation_config.assert_zero_decision_delay is True


def test_initial_video_model_chunk_starts_after_initial_frame() -> None:
    """The USDZ initial frame is at render start; generated frames follow it."""

    render_start_us = 1_000_000
    frame_interval_us = 33_333
    renderer_service = SimpleNamespace(
        chunk_size=5,
        frame_interval_us=frame_interval_us,
        config=SimpleNamespace(chunk_frames=8),
    )

    event = make_initial_video_model_render_event(
        scene_start_us=render_start_us,
        control_timestep_us=8 * frame_interval_us,
        runtime_cameras=[],
        renderer_service=renderer_service,
        driver=SimpleNamespace(),
        broadcaster=SimpleNamespace(),
    )

    assert isinstance(event, VideoModelPrefetchEvent)
    assert event.timestamp_us == render_start_us
    assert event.chunk_start_us == render_start_us + frame_interval_us
    assert event.chunk_size == 5

    renderer_service.chunk_size = 8
    queue = EventQueue()
    event._schedule_next_prefetch(queue, frame_interval_us)
    next_event = queue.pop()

    assert isinstance(next_event, VideoModelPrefetchEvent)
    assert next_event.timestamp_us == render_start_us + 5 * frame_interval_us
    assert next_event.chunk_start_us == render_start_us + 6 * frame_interval_us
    assert next_event.chunk_size == 8


def test_initial_video_model_event_validates_control_timestep() -> None:
    renderer_service = SimpleNamespace(
        chunk_size=5,
        frame_interval_us=33_333,
        config=SimpleNamespace(chunk_frames=8),
    )

    with pytest.raises(ValueError, match="control_timestep_us"):
        make_initial_video_model_render_event(
            scene_start_us=1_000_000,
            control_timestep_us=100_000,
            runtime_cameras=[],
            renderer_service=renderer_service,
            driver=SimpleNamespace(),
            broadcaster=SimpleNamespace(),
        )
