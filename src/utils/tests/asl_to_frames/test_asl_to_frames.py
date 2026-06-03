# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import numpy as np
import pytest
from alpasim_grpc.v0.common_pb2 import Pose, PoseAtTime, Quat, Trajectory, Vec3
from alpasim_grpc.v0.egodriver_pb2 import DriveSessionRequest
from alpasim_grpc.v0.logging_pb2 import LogEntry, RolloutMetadata
from alpasim_grpc.v0.video_model_pb2 import (
    CameraOutput,
    Image,
    VideoChunkRequest,
    VideoChunkReturn,
)
from alpasim_utils.asl_to_frames.__main__ import (
    Frame,
    convert_single_log,
    determine_save_dir,
    save_frames_as_files,
)


def test_determine_save_dir():
    log_path = "mnt/rollouts/cliggt-hash/rollout.asl"

    # nominal path, log_save_dir unspecified
    save_dir = determine_save_dir(log_path, None)
    expected_save_dir = "mnt/rollouts/cliggt-hash/rollout_asl_frames"
    assert save_dir == expected_save_dir, f"{save_dir=} {expected_save_dir=}"

    # path required by kpi, log_save_dir specified
    save_dir = determine_save_dir(log_path, "mnt/outputs")
    expected_save_dir = "mnt/outputs/rollouts/cliggt-hash/rollout"
    assert save_dir == expected_save_dir, f"{save_dir=} {expected_save_dir=}"


@pytest.mark.asyncio
async def test_save_frames_as_files_uses_supplied_timestamps(monkeypatch, tmp_path):
    written_paths = []

    async def fake_write_image(content: bytes, path: str) -> None:
        del content
        written_paths.append(path)

    monkeypatch.setattr(
        "alpasim_utils.asl_to_frames.__main__._write_image", fake_write_image
    )

    images = [
        Frame(image_bytes=b"first", timestamp_us=1),
        Frame(image_bytes=b"second", timestamp_us=2),
    ]
    await save_frames_as_files(
        images,
        np.array([200, 100], dtype=np.uint64),
        str(tmp_path),
    )

    assert sorted(written_paths) == [
        f"{tmp_path}/100",
        f"{tmp_path}/200",
    ]


@pytest.mark.asyncio
async def test_convert_single_log_writes_video_model_rgb_and_hdmap_mp4s(
    monkeypatch, tmp_path
):
    written_mp4s = []

    async def fake_read_pb_log(_path: str):
        for entry in _video_model_log_entries():
            yield entry

    async def fake_frames_to_mp4(images, timestamps_us, save_path):
        written_mp4s.append(
            (
                save_path,
                [image.image_bytes for image in images],
                timestamps_us.tolist(),
            )
        )

    monkeypatch.setattr(
        "alpasim_utils.asl_to_frames.__main__.async_read_pb_log", fake_read_pb_log
    )
    monkeypatch.setattr(
        "alpasim_utils.asl_to_frames.__main__.frames_to_mp4", fake_frames_to_mp4
    )

    await convert_single_log("rollout.asl", str(tmp_path), "mp4")

    assert written_mp4s == [
        (
            f"{tmp_path}/video_model_rgb_camera_front",
            [b"rgb-100", b"rgb-200"],
            [100, 200],
        ),
        (
            f"{tmp_path}/video_model_hdmap_camera_front",
            [b"hdmap-100", b"hdmap-200"],
            [100, 200],
        ),
    ]


def _video_model_log_entries() -> list[LogEntry]:
    return [
        LogEntry(rollout_metadata=RolloutMetadata()),
        LogEntry(driver_session_request=DriveSessionRequest()),
        LogEntry(
            video_model_chunk_request=VideoChunkRequest(
                rig_trajectory=_trajectory([100, 200])
            )
        ),
        LogEntry(
            video_model_chunk_return=VideoChunkReturn(
                camera_outputs=[
                    CameraOutput(
                        camera_logical_id="camera_front",
                        rgb_frames=[
                            Image(data=b"rgb-100"),
                            Image(data=b"rgb-200"),
                        ],
                        hdmap_condition_frames=[
                            Image(data=b"hdmap-100"),
                            Image(data=b"hdmap-200"),
                        ],
                    )
                ]
            )
        ),
    ]


def _trajectory(timestamps_us: list[int]) -> Trajectory:
    return Trajectory(poses=[_pose_at(timestamp_us) for timestamp_us in timestamps_us])


def _pose_at(timestamp_us: int) -> PoseAtTime:
    return PoseAtTime(
        timestamp_us=timestamp_us,
        pose=Pose(
            vec=Vec3(x=0.0, y=0.0, z=0.0),
            quat=Quat(w=1.0, x=0.0, y=0.0, z=0.0),
        ),
    )
