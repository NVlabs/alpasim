# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Implements methods for reading and writing logs as length-delimited protobuf
files.
See https://seb-nyberg.medium.com/length-delimited-protobuf-streams-a39ebc4a4565
for explanation of the format and rationale.
"""

import logging
import os
import struct
from dataclasses import dataclass
from typing import AsyncGenerator, Optional, Self, Type, TypeVar

import aiofiles
import aiofiles.os
import numpy as np
from alpasim_grpc.v0.logging_pb2 import ActorPoses, LogEntry, RolloutMetadata
from alpasim_utils.geometry import Trajectory, pose_from_grpc
from google.protobuf.message import Message

logger = logging.getLogger(__name__)


@dataclass
class LogWriter:
    """
    Class for writing protobuf logs.
    The current implementation is just a wrapper around an open file but in the future
    we may exceed the OS limit of 4096 open files and want to rework this class to keep
    logs in memory and only open files to write periodically.
    """

    file_path: str
    file_handle: Optional[aiofiles.threadpool.binary.AsyncBufferedIOBase] = None

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        if not self.file_path:
            raise ValueError("Log file path must be non-empty.")

    async def __aenter__(self) -> Self:
        file_dir, file_name = os.path.split(self.file_path)
        if file_dir:
            await aiofiles.os.makedirs(file_dir, exist_ok=True)
        self.file_handle = await aiofiles.open(self.file_path, "wb")
        return self

    async def __aexit__(self, *args, **kwargs) -> None:
        if self.file_handle is None:
            raise AssertionError(
                "__aexit__ called with self.file_handle == None (should be initialized in __aenter__)."
            )
        await self.file_handle.close()
        self.file_handle = None

    async def on_message(self, message: LogEntry) -> None:
        """Handle a LogEntry message by writing it to the log file."""
        if self.file_handle is None:
            raise RuntimeError(
                "Using LogWriter.on_message outside of `async with log_writer`."
            )
        await write_pb_stream(self.file_handle, message)


async def write_pb_stream(
    file: aiofiles.threadpool.binary.AsyncBufferedIOBase, message: Message
) -> None:
    message_binary = message.SerializeToString()
    size_prefix = struct.pack(">L", len(message_binary))
    await file.write(size_prefix + message_binary)


M = TypeVar("M", bound=Message)


async def async_read_pb_stream(
    fname: str, message_type: Type[M], raise_on_malformed: bool = False
) -> AsyncGenerator[M, None]:
    async with aiofiles.open(fname, "rb") as file:
        while (size_prefix_chunk := await file.read(4)) != b"":  # detect EOF
            (message_size,) = struct.unpack(">L", size_prefix_chunk)
            message_chunk = await file.read(message_size)
            if len(message_chunk) != message_size:
                error = f"Malformed file (expected {message_size} bytes, found {len(message_chunk)})"
                if raise_on_malformed:
                    raise IOError(error)
                else:
                    logger.warning(error)
                    break
            else:
                message = message_type.FromString(message_chunk)
                yield message


async def async_read_pb_log(
    fname: str, raise_on_malformed: bool = False
) -> AsyncGenerator[LogEntry, None]:
    async for log_entry in async_read_pb_stream(
        fname, message_type=LogEntry, raise_on_malformed=raise_on_malformed
    ):
        yield log_entry


async def read_trajectory(fname: str) -> Optional[tuple[str, Trajectory]]:
    """
    Read a log stream, select actor poses and combine in a trajectory object.
    Return scene name + trajectory
    """
    timestamps_us = []
    poses = []

    name: Optional[str] = None

    async for message in async_read_pb_log(fname):
        if message.WhichOneof("log_entry") == "rollout_metadata":
            metadata: RolloutMetadata = message.rollout_metadata
            name = metadata.session_metadata.scene_id
            continue
        if message.WhichOneof("log_entry") != "actor_poses":
            continue
        poses_message: ActorPoses = message.actor_poses
        timestamp_us = poses_message.timestamp_us
        (grpc_pose,) = poses_message.actor_poses
        pose = pose_from_grpc(grpc_pose.actor_pose)
        timestamps_us.append(timestamp_us)
        poses.append(pose)

    if not timestamps_us:
        return None
    if not name:
        return None

    trajectory = Trajectory.from_poses(
        timestamps=np.array(timestamps_us, dtype=np.uint64),
        poses=poses,  # List of Pose objects
    )

    return name, trajectory
