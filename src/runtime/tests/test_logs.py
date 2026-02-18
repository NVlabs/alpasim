# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

from tempfile import TemporaryDirectory

from alpasim_grpc.v0.logging_pb2 import LogEntry, RolloutMetadata
from alpasim_runtime.broadcaster import MessageBroadcaster
from alpasim_utils.logs import LogWriter, async_read_pb_log


async def test_log_is_written() -> None:
    """Checks that writing an example message to LogWriter creates an .asl file which can be decoded successfully"""
    written_message = LogEntry(
        rollout_metadata=RolloutMetadata(
            rollout_index=3,
            # leave the other fields to defaults
        )
    )

    with TemporaryDirectory() as temp_dir:
        asl_path = f"{temp_dir}/log.asl"

        asl_log_writer = LogWriter(file_path=asl_path)

        broadcaster = MessageBroadcaster(handlers=[asl_log_writer])
        async with broadcaster:
            await broadcaster.broadcast(message=written_message)

        n_messages = 0
        async for read_message in async_read_pb_log(asl_path, raise_on_malformed=True):
            n_messages += 1

        assert n_messages == 1, "read more messages than were written"
        assert read_message == written_message
