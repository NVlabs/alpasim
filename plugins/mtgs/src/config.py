# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""MTGS renderer plugin configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MTGSConfig:
    """Configuration for MTGS renderer plugin (client-side).

    Since the MTGS server implements the standard SensorsimService gRPC
    interface, the client behavior is nearly identical to the built-in
    sensorsim client. This config only exposes MTGS-specific knobs.
    """

    skip_warmup: bool = False
