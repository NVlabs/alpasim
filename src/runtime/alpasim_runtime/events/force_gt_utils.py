# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Helpers for force-GT event behavior."""

from __future__ import annotations

from alpasim_utils import geometry

FORCE_GT_REFERENCE_HORIZON_US = 5_000_000


def controller_reference_trajectory(
    force_gt_trajectory: geometry.Trajectory, step_start_us: int
) -> geometry.Trajectory:
    """Build the controller reference used while the rollout is force-GT driven."""
    clip_end_us = min(
        step_start_us + FORCE_GT_REFERENCE_HORIZON_US + 1,
        force_gt_trajectory.time_range_us.stop,
    )
    return force_gt_trajectory.clip(step_start_us, clip_end_us)
