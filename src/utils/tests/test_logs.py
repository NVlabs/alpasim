# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import alpasim_utils.logs as logs
import pytest
import pytest_asyncio
from alpasim_grpc.v0.logging_pb2 import ActorPoses, LogEntry, RolloutMetadata


@pytest.fixture
def valid_actor_poses() -> ActorPoses:
    actor_poses = ActorPoses()
    actor_poses.timestamp_us = 1234
    actor_pose_0 = actor_poses.actor_poses.add()
    actor_pose_0.actor_id = "EGO"
    actor_pose_0.actor_pose.vec.x = 1.0
    actor_pose_0.actor_pose.vec.y = 2.0
    actor_pose_0.actor_pose.vec.z = 3.0
    actor_pose_0.actor_pose.quat.w = 1.0

    return actor_poses


@pytest.mark.asyncio
async def test_LogWriter_error_on_invalid_file_name():
    with pytest.raises(ValueError):
        logs.LogWriter("")


@pytest.mark.asyncio
async def test_LogWriter_error_on_invalid_file_handle(tmp_path, valid_actor_poses):
    log_writer = logs.LogWriter(tmp_path / "test_log.asl")
    with pytest.raises(RuntimeError):
        await log_writer.on_message(LogEntry(actor_poses=valid_actor_poses))


@pytest.mark.asyncio
async def test_LogWriter_write(tmp_path, valid_actor_poses):
    log_writer = logs.LogWriter(tmp_path / "test_log.asl")
    async with log_writer:
        await log_writer.on_message(LogEntry(actor_poses=valid_actor_poses))

    # verify that the data was written
    with open(tmp_path / "test_log.asl", "rb") as f:
        data = f.read()
        log_entry = LogEntry()
        # note that the first 4 bytes are the size of the message. This seems like
        # something that might be revisited later, e.g. to move to a magic number
        # separation, which would be more robust to dropouts/corrpution.
        log_entry.ParseFromString(data[4:])
        assert log_entry.actor_poses.timestamp_us == valid_actor_poses.timestamp_us
        assert (
            log_entry.actor_poses.actor_poses[0].actor_id
            == valid_actor_poses.actor_poses[0].actor_id
        )


@pytest_asyncio.fixture
async def sample_log_file(tmp_path, valid_actor_poses):
    file_path = tmp_path / "sample_log.asl"
    log_writer = logs.LogWriter(file_path)
    async with log_writer:
        # write out the session metadata
        metadata = RolloutMetadata()
        metadata.session_metadata.scene_id = "test_scene"
        await log_writer.on_message(LogEntry(rollout_metadata=metadata))

        # write out a handful of actor poses
        actor_poses = ActorPoses()
        actor_poses.CopyFrom(valid_actor_poses)
        for timestamp in range(0, 100000, 10000):
            actor_poses.timestamp_us = timestamp
            actor_poses.actor_poses[0].actor_pose.vec.x = float(timestamp) / 1e6
            await log_writer.on_message(LogEntry(actor_poses=actor_poses))
    return file_path


@pytest.mark.asyncio
async def test_async_read_pb_stream(sample_log_file):
    # read the sample log file, and perform some sanity checks
    message_count = 0
    async for message in logs.async_read_pb_stream(sample_log_file, LogEntry):
        message_count += 1
        assert message.WhichOneof("log_entry") in ["actor_poses", "rollout_metadata"]
    assert message_count == 11  # 1 metadata + 10 actor poses


@pytest.mark.asyncio
async def test_async_read_pb_log(sample_log_file):
    # read the sample log file, and perform some sanity checks
    message_count = 0
    async for message in logs.async_read_pb_log(sample_log_file):
        message_count += 1
        assert message.WhichOneof("log_entry") in ["actor_poses", "rollout_metadata"]
    assert message_count == 11  # 1 metadata + 10 actor poses
