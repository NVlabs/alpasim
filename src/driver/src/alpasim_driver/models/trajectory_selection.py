# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Pick one trajectory out of several sampled candidates.

Sampling multiple trajectories per planning cycle and always taking the first
one makes the driven path jump between modes (e.g. "keep lane" on one cycle,
"change lane" on the next).  The strategies here keep the driven path
consistent by scoring candidates against the plan selected in the previous
cycle.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from alpasim_utils.geometry import Pose, Trajectory

# Scoring against the previous plan needs a meaningful stretch of it to still
# lie ahead of the current planning time.
MIN_OVERLAP_STEPS = 4


class TrajectorySelectionStrategy(StrEnum):
    """Strategy for picking one candidate trajectory."""

    # Take the first sample, i.e. no selection.
    ALWAYS_FIRST = "ALWAYS_FIRST"
    # Smallest mean 3D distance to the previous plan.
    CLOSEST_3D = "CLOSEST_3D"
    # Smallest mean lateral distance to the previous plan.
    CLOSEST_LATERAL = "CLOSEST_LATERAL"


def select_trajectory(
    *,
    candidate_positions: np.ndarray,
    candidate_timestamps_us: np.ndarray,
    previous_plan_in_local: Trajectory | None,
    pose_local_to_rig_t0: Pose,
    strategy: TrajectorySelectionStrategy,
    max_num_distance_points: int,
    skip_first_n_distance_points: int,
) -> int:
    """Return the index of the candidate to drive.

    Args:
        candidate_positions: Candidate waypoints of shape ``(K, T, 3)`` in the
            rig frame at t0.
        candidate_timestamps_us: Waypoint timestamps of shape ``(T,)``, strictly
            increasing.
        previous_plan_in_local: Plan selected in the previous cycle, in the local
            frame, or None on the first cycle of a session.
        pose_local_to_rig_t0: Pose of the rig at t0 in the local frame.
        strategy: Selection strategy.
        max_num_distance_points: Number of waypoints entering the distance
            average, counted from ``skip_first_n_distance_points``.
        skip_first_n_distance_points: Number of leading waypoints excluded from
            the distance average.  The waypoints right after t0 barely differ
            between candidates, so skipping them sharpens the comparison.

    Returns:
        Index into the first axis of ``candidate_positions``.

    Raises:
        ValueError: If the previous plan does not overlap the candidates for at
            least :data:`MIN_OVERLAP_STEPS` waypoints.
    """
    if strategy is TrajectorySelectionStrategy.ALWAYS_FIRST:
        return 0

    if previous_plan_in_local is None:
        # First cycle of the session: there is nothing to stay consistent with.
        return 0

    previous_timestamps_us = previous_plan_in_local.timestamps_us
    if candidate_timestamps_us[0] < previous_timestamps_us[0]:
        raise ValueError(
            f"Candidate trajectory starts at {candidate_timestamps_us[0]}us, before "
            f"the previous plan at {previous_timestamps_us[0]}us; planning time must "
            "advance."
        )

    # Timestamps are increasing, so the overlap with the previous plan is a
    # prefix of the candidate waypoints.
    num_overlap = int(
        np.count_nonzero(candidate_timestamps_us <= previous_timestamps_us[-1])
    )
    if num_overlap < MIN_OVERLAP_STEPS:
        raise ValueError(
            f"Previous plan overlaps the current candidates for {num_overlap} "
            f"waypoint(s), need at least {MIN_OVERLAP_STEPS}. This happens when "
            "planning is triggered less frequently than the prediction horizon."
        )

    # Both sides of the comparison end up in the rig frame at t0, so the scores
    # are distances between two predictions of the same stretch of road.  Framing
    # the previous plan with a pose that trails t0 instead would leak part of the
    # ego motion between the cycles into every candidate's score.
    previous_positions = (
        previous_plan_in_local.interpolate(candidate_timestamps_us[:num_overlap])
        .transform(pose_local_to_rig_t0.inverse())
        .positions
    )
    candidates = candidate_positions[:, :num_overlap]

    if strategy is TrajectorySelectionStrategy.CLOSEST_3D:
        distances = np.linalg.norm(candidates - previous_positions, axis=-1)
    elif strategy is TrajectorySelectionStrategy.CLOSEST_LATERAL:
        # Lateral distance is approximated by the y offset in the rig frame at t0.
        distances = np.abs(candidates[:, :, 1] - previous_positions[:, 1])
    else:
        raise NotImplementedError(strategy)

    start = min(skip_first_n_distance_points, distances.shape[1] - 1)
    end = min(start + max_num_distance_points, distances.shape[1])
    return int(distances[:, start:end].mean(axis=1).argmin())


def plan_in_local_frame(
    positions: np.ndarray,
    rotations: np.ndarray,
    timestamps_us: np.ndarray,
    pose_local_to_rig_t0: Pose,
) -> Trajectory:
    """Express a selected plan in the local frame.

    The local frame outlives the planning cycle, so a plan stored this way can
    be compared against the candidates of the next cycle.

    Args:
        positions: Waypoints of shape ``(T, 3)`` in the rig frame at t0.
        rotations: Waypoint rotation matrices of shape ``(T, 3, 3)`` in the rig
            frame at t0.
        timestamps_us: Waypoint timestamps of shape ``(T,)``.
        pose_local_to_rig_t0: Pose of the rig at t0 in the local frame.

    Returns:
        The plan as a trajectory in the local frame.
    """
    se3 = np.tile(np.eye(4, dtype=np.float32), (len(positions), 1, 1))
    se3[:, :3, :3] = rotations
    se3[:, :3, 3] = positions
    poses = [Pose.from_se3(waypoint) for waypoint in se3]
    return Trajectory.from_poses(timestamps_us, poses).transform(pose_local_to_rig_t0)
