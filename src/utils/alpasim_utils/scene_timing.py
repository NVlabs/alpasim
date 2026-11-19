# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Shared feasibility checks for recording-relative scene timing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from alpasim_utils.types import T0_AUTO


class SceneTimingRejectionReason(str, Enum):
    """Reasons an explicit policy start cannot execute under a simulation config."""

    INSUFFICIENT_HISTORY = "insufficient_history"
    INSUFFICIENT_HORIZON = "insufficient_horizon"


@dataclass(frozen=True)
class SceneTimingEvaluation:
    """Result of evaluating one policy start against one clip duration."""

    rejection_reason: SceneTimingRejectionReason | None
    clip_start_offset_us: int | None
    requested_end_us: int | None

    @property
    def accepted(self) -> bool:
        """Return whether the scene can execute under the timing config."""
        return self.rejection_reason is None


def derive_clip_start_offset_us(
    policy_start_us: int | Literal["AUTO"],
    force_gt_duration_us: int,
) -> int | None:
    """Derive the recording-relative rollout start for an authored policy start.

    Returns:
        The clip offset in microseconds, or ``None`` for AUTO timing.

    Raises:
        ValueError: An explicit policy start cannot supply the forced history.
    """
    if policy_start_us == T0_AUTO:
        return None
    if policy_start_us < force_gt_duration_us:
        raise ValueError(
            f"policy_start_us={policy_start_us} does not provide "
            f"force_gt_duration_us={force_gt_duration_us} of history"
        )
    return policy_start_us - force_gt_duration_us


def evaluate_scene_timing(
    *,
    policy_start_us: int | Literal["AUTO"],
    force_gt_duration_us: int,
    n_sim_steps: int,
    control_timestep_us: int,
    clip_duration_us: int,
) -> SceneTimingEvaluation:
    """Evaluate an authored policy start against one clip and simulation config.

    AUTO timing is accepted without recording-bound checks. For explicit T0,
    the result records insufficient history or horizon without mutating the
    authored value.
    """
    try:
        clip_start_offset_us = derive_clip_start_offset_us(
            policy_start_us,
            force_gt_duration_us,
        )
    except ValueError:
        return SceneTimingEvaluation(
            rejection_reason=SceneTimingRejectionReason.INSUFFICIENT_HISTORY,
            clip_start_offset_us=None,
            requested_end_us=None,
        )

    if clip_start_offset_us is None:
        return SceneTimingEvaluation(
            rejection_reason=None,
            clip_start_offset_us=None,
            requested_end_us=None,
        )

    requested_end_us = clip_start_offset_us + n_sim_steps * control_timestep_us
    rejection_reason = (
        SceneTimingRejectionReason.INSUFFICIENT_HORIZON
        if requested_end_us > clip_duration_us
        else None
    )
    return SceneTimingEvaluation(
        rejection_reason=rejection_reason,
        clip_start_offset_us=clip_start_offset_us,
        requested_end_us=requested_end_us,
    )
