# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import argparse
import asyncio
from typing import Optional

from alpasim_grpc.v0.logging_pb2 import LogEntry
from alpasim_utils.logs import async_read_pb_log


async def print_asl(
    file_path: str,
    start: int,
    end: Optional[int],
    message_types: set[str],
    just_types: bool,
) -> None:
    message_i = 0
    async for log_entry in async_read_pb_log(file_path):
        if message_i == end:
            break

        if message_i < start:
            continue

        if log_entry.WhichOneof("log_entry") in message_types:
            if just_types:
                print(log_entry.WhichOneof("log_entry"))
            else:
                if log_entry.WhichOneof("log_entry") == "driver_camera_image":
                    log_entry.driver_camera_image.camera_image.image_bytes = (
                        b"<image data redacted>"
                    )
                print(log_entry)

        message_i += 1


log_entry_fields: tuple[str] = tuple(field.name for field in LogEntry.DESCRIPTOR.fields)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "asl_file",
        type=str,
        help="Path of the .asl file to print",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Index of the first message to print (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Index of the last message to print (default: print all)",
    )
    parser.add_argument(
        "--message-types",
        nargs="+",
        choices=log_entry_fields,
        default=log_entry_fields,
        metavar="MSG_TYPE",
        help=f"Message types to print, by default all of {', '.join(log_entry_fields)}.",
    )
    parser.add_argument(
        "--just-types",
        action="store_true",
        help="Only print the type of the message (and not the content)",
    )
    args = parser.parse_args()

    asyncio.run(
        print_asl(
            file_path=args.asl_file,
            start=args.start,
            end=args.end,
            message_types=frozenset(args.message_types),
            just_types=args.just_types,
        )
    )
