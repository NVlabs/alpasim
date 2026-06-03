# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Helpers for force-GT physics blending."""

from __future__ import annotations

from alpasim_runtime.unbound_rollout import UnboundRollout


def force_gt_physics_blend_hold_end_us(unbound: UnboundRollout) -> int:
    """Timestamp through which recorded GT should be held exactly.

    The blend begins where rendering begins — at the first GT camera frame's
    shutter close — and runs through the end of the force-GT period.
    """
    return unbound.render_start_timestamp_us


def force_gt_physics_blend_alpha(unbound: UnboundRollout, timestamp_us: int) -> float:
    """Blend alpha from recorded GT (0.0) to physics-corrected pose (1.0)."""
    gt_hold_end_us = force_gt_physics_blend_hold_end_us(unbound)
    blend_end_us = unbound.render_start_timestamp_us + unbound.force_gt_duration_us

    if blend_end_us <= gt_hold_end_us:
        return 0.0

    return max(
        0.0,
        min(1.0, (timestamp_us - gt_hold_end_us) / (blend_end_us - gt_hold_end_us)),
    )
