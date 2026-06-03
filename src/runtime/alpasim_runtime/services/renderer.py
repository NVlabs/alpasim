# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Renderer service protocol used by rollout preparation and event execution."""

from __future__ import annotations

from typing import Any, Protocol

from alpasim_runtime.broadcaster import MessageBroadcaster
from alpasim_runtime.config import SimulationConfig


class RendererService(Protocol):
    """Protocol for the active renderer service.

    Renderer implementations own render-event creation and the timing
    constraints that make a rollout valid for their rendering cadence.
    """

    def make_initial_render_event(self, **kwargs: Any) -> Any:
        """Create the renderer's initial event or events."""
        ...

    def validate_timing_alignment(self, simulation_config: SimulationConfig) -> None:
        """Raise if the simulation timing is invalid for this renderer."""
        ...

    def required_policy_start_timestmap_us(
        self,
        render_start_timestamp_us: int,
    ) -> int:
        """Return the policy start timestamp required by this renderer.

        Rollout construction uses this value as part of validation and
        scheduling; if the returned timestamp cannot be supported by the
        recording or simulation bounds, the rollout is invalid.
        """
        ...

    def rollout_session(
        self,
        uuid: str,
        broadcaster: MessageBroadcaster,
        session_config: object | None = None,
    ) -> Any:
        """Return the renderer rollout-session async context manager."""
        ...
