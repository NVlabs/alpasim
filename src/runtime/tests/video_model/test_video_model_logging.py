# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import pytest
from alpasim_grpc.v0.common_pb2 import VersionId
from alpasim_grpc.v0.logging_pb2 import LogEntry, RolloutMetadata
from alpasim_grpc.v0.video_model_pb2 import (
    CameraOutput,
    Image,
    ImageFormat,
    SessionCloseRequest,
    SessionId,
    SessionRequest,
    StaticWorldMap,
    VideoChunkRequest,
    VideoChunkReturn,
)
from alpasim_runtime.replay_services.asl_reader import ASLReader
from alpasim_utils.logs import LogWriter
from alpasim_utils.print_asl.__main__ import print_asl


@pytest.mark.asyncio
async def test_asl_reader_pairs_video_model_exchanges(tmp_path) -> None:
    asl_path = tmp_path / "video-model.asl"
    async with LogWriter(str(asl_path)) as writer:
        await writer.on_message(
            LogEntry(
                video_model_session_request=SessionRequest(
                    static_world_map=StaticWorldMap(hdmap_parquets=b"map")
                )
            )
        )
        await writer.on_message(
            LogEntry(video_model_session_id=SessionId(session_id="session-1"))
        )
        await writer.on_message(
            LogEntry(
                video_model_chunk_request=VideoChunkRequest(
                    session_id=SessionId(session_id="session-1")
                )
            )
        )
        await writer.on_message(
            LogEntry(
                video_model_chunk_return=VideoChunkReturn(
                    camera_outputs=[
                        CameraOutput(
                            camera_logical_id="camera_front_wide_120fov",
                            rgb_frames=[
                                Image(data=b"rgb-bytes", format=ImageFormat.JPEG)
                            ],
                        )
                    ]
                )
            )
        )
        await writer.on_message(
            LogEntry(
                video_model_session_close_request=SessionCloseRequest(
                    session_id="session-1"
                )
            )
        )

    reader = ASLReader(str(asl_path))
    await reader.load_exchanges()

    assert len(reader.get_exchanges("video_model", "start_session")) == 1
    assert len(reader.get_exchanges("video_model", "render_video_chunk")) == 1
    assert len(reader.get_exchanges("video_model", "close_session")) == 1


def test_asl_reader_returns_video_model_version_from_metadata() -> None:
    reader = ASLReader("dummy.asl")
    reader.asl_metadata["rollout_metadata"] = RolloutMetadata(
        version_ids=RolloutMetadata.VersionIds(
            video_model_version=VersionId(version_id="omnidreams", git_hash="abc123")
        )
    )

    version = reader.get_service_version("video_model")

    assert version is not None
    assert version.version_id == "omnidreams"
    assert version.git_hash == "abc123"


@pytest.mark.asyncio
async def test_print_asl_redacts_video_model_payloads(tmp_path, capsys) -> None:
    asl_path = tmp_path / "video-model.asl"
    async with LogWriter(str(asl_path)) as writer:
        await writer.on_message(
            LogEntry(
                video_model_session_request=SessionRequest(
                    static_world_map=StaticWorldMap(hdmap_parquets=b"secret-map"),
                    initial_frames=[Image(data=b"secret-initial")],
                )
            )
        )
        await writer.on_message(
            LogEntry(
                video_model_chunk_return=VideoChunkReturn(
                    camera_outputs=[
                        CameraOutput(
                            rgb_frames=[Image(data=b"secret-rgb")],
                            hdmap_condition_frames=[Image(data=b"secret-hdmap")],
                        )
                    ],
                )
            )
        )

    await print_asl(
        file_path=str(asl_path),
        start=0,
        end=None,
        message_types={"video_model_session_request", "video_model_chunk_return"},
        just_types=False,
    )

    printed = capsys.readouterr().out
    assert "secret" not in printed
    assert "<image data redacted>" in printed
    assert "<hdmap data redacted>" in printed
