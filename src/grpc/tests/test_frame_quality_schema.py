# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for the frame-quality RPC and persisted log schema."""

from alpasim_grpc.v0 import frame_quality_pb2, logging_pb2


def test_frame_quality_response_round_trips_through_log_entry() -> None:
    entry = logging_pb2.LogEntry(
        frame_quality_response=frame_quality_pb2.FrameQualityResponse(
            session_uuid="rollout",
            camera_logical_id="front",
            timestamp_us=123,
            score=0.75,
            checkpoint_id="v1",
            checkpoint_sha256="abc",
        )
    )

    restored = logging_pb2.LogEntry.FromString(entry.SerializeToString())

    assert restored.WhichOneof("log_entry") == "frame_quality_response"
    assert restored.frame_quality_response == entry.frame_quality_response


def test_frame_quality_contract_scores_one_frame_per_rpc() -> None:
    service = frame_quality_pb2.DESCRIPTOR.services_by_name["FrameQualityService"]

    assert set(service.methods_by_name) == {"get_version", "score_frame"}
    method = service.methods_by_name["score_frame"]
    assert method.input_type.name == "FrameQualityRequest"
    assert method.output_type.name == "FrameQualityResponse"

    request = frame_quality_pb2.FrameQualityRequest(
        session_uuid="rollout",
        camera_logical_id="front",
        timestamp_us=123,
        image_bytes=b"jpeg",
    )
    response = frame_quality_pb2.FrameQualityResponse(
        session_uuid="rollout",
        camera_logical_id="front",
        timestamp_us=123,
        score=0.75,
        checkpoint_id="v1",
        checkpoint_sha256="abc",
    )

    assert request.session_uuid == response.session_uuid
    assert request.camera_logical_id == response.camera_logical_id
    assert request.timestamp_us == response.timestamp_us
