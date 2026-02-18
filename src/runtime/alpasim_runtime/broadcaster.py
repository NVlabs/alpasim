# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Message broadcasting infrastructure for the runtime simulation loop.

This module provides the MessageBroadcaster class that dispatches LogEntry
messages to all registered handlers. Each handler is responsible for its
own message processing/decoding logic.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Protocol, Self

from alpasim_grpc.v0.logging_pb2 import LogEntry


class MessageHandler(Protocol):
    """
    Protocol for message handlers.

    All message consumers (LogWriter, RCLogLogWriter, RuntimeEvaluator) implement
    this interface. Each handler is responsible for processing the messages it
    cares about and ignoring others.
    """

    async def on_message(self, message: LogEntry) -> None:
        """
        Handle a LogEntry message.

        Args:
            message: The LogEntry protobuf message to process.
        """
        ...


@dataclass
class MessageBroadcaster:
    """
    Broadcasts LogEntry messages to all registered handlers.

    This is a simple dispatcher that iterates through handlers and calls
    on_message() on each. Each handler is responsible for its own message
    processing logic.

    Usage:
        broadcaster = MessageBroadcaster(
            handlers=[asl_writer, rclog_writer, runtime_evaluator]
        )

        async with broadcaster:
            await broadcaster.broadcast(LogEntry(driver_camera_image=...))
    """

    handlers: list[MessageHandler] = field(default_factory=list)

    async def broadcast(self, message: LogEntry) -> None:
        """
        Broadcast a LogEntry message to all handlers concurrently.

        Args:
            message: The LogEntry protobuf message to broadcast.
        """
        if not self.handlers:
            return

        await asyncio.gather(*[h.on_message(message) for h in self.handlers])

    async def __aenter__(self) -> Self:
        """Enter async context - initialize all handlers that support it."""
        for handler in self.handlers:
            if hasattr(handler, "__aenter__"):
                await handler.__aenter__()
        return self

    async def __aexit__(self, *args, **kwargs) -> None:
        """Exit async context - cleanup all handlers that support it."""
        for handler in self.handlers:
            if hasattr(handler, "__aexit__"):
                await handler.__aexit__(*args, **kwargs)
