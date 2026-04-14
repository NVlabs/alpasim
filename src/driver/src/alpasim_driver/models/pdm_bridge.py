# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Minimal dependency-free PDM-style planning bridge for driver integration.

This module intentionally does not depend on navsim / nuplan planner APIs.
It consumes the compact planner_context built by runtime PolicyEvent and
produces a closed-loop-scored trajectory in rig frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from .base import DriveCommand


@dataclass(frozen=True, slots=True)
class PDMEgoState:
    speed_mps: float
    acceleration_mps2: float
    yaw_rad: float


@dataclass(frozen=True, slots=True)
class PDMActor:
    actor_id: str
    position_in_rig: np.ndarray
    yaw_rad: float


@dataclass(frozen=True, slots=True)
class PDMWaitLine:
    points_in_rig: np.ndarray


@dataclass(frozen=True, slots=True)
class PDMPlannerInput:
    ego: PDMEgoState
    route_waypoints_in_rig: np.ndarray
    nearby_lane_centerlines_in_rig: list[np.ndarray]
    actors: list[PDMActor]
    wait_lines_in_rig: list[PDMWaitLine]
    crosswalks_in_rig: list[np.ndarray]
    fallback_command: DriveCommand


@dataclass(frozen=True, slots=True)
class PDMClosedLoopResult:
    trajectory_xy: np.ndarray
    headings: np.ndarray
    debug_metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _ProposalCandidate:
    proposal_idx: int
    source: str
    centerline_xy: np.ndarray
    trajectory_xy: np.ndarray
    headings: np.ndarray
    score: float
    score_breakdown: dict[str, float]


def build_pdm_planner_input(
    *,
    speed_mps: float,
    acceleration_mps2: float,
    planner_context: dict[str, Any] | None,
    fallback_command: DriveCommand,
) -> PDMPlannerInput:
    planner_context = planner_context or {}

    ego_ctx = planner_context.get("ego")
    ego_yaw = 0.0
    if isinstance(ego_ctx, dict):
        ego_yaw = float(ego_ctx.get("yaw", 0.0))

    route_waypoints = _points_from_context(
        planner_context.get("route_waypoints_in_rig"),
        minimum_cols=2,
    )
    nearby_lanes_ctx = planner_context.get("nearby_lanes")
    nearby_lane_centerlines: list[np.ndarray] = []
    if isinstance(nearby_lanes_ctx, list):
        for lane in nearby_lanes_ctx:
            if not isinstance(lane, dict):
                continue
            points = _points_from_context(lane.get("centerline_in_rig"), minimum_cols=2)
            if len(points) >= 2:
                nearby_lane_centerlines.append(points)

    actors_ctx = planner_context.get("actors")
    actors: list[PDMActor] = []
    if isinstance(actors_ctx, list):
        for actor in actors_ctx:
            if not isinstance(actor, dict):
                continue
            pos = actor.get("position_in_rig")
            if not isinstance(pos, list) or len(pos) < 2:
                continue
            actors.append(
                PDMActor(
                    actor_id=str(actor.get("id", "")),
                    position_in_rig=np.asarray(pos[:2], dtype=np.float64),
                    yaw_rad=float(actor.get("yaw", 0.0)),
                )
            )

    traffic_rules = planner_context.get("traffic_rules")
    wait_lines: list[PDMWaitLine] = []
    crosswalks: list[np.ndarray] = []
    if isinstance(traffic_rules, dict):
        for wait_line in traffic_rules.get("wait_lines_in_rig", []):
            if not isinstance(wait_line, dict):
                continue
            points = _points_from_context(wait_line.get("points"), minimum_cols=2)
            if len(points) >= 2:
                wait_lines.append(PDMWaitLine(points_in_rig=points))
        for crosswalk in traffic_rules.get("crosswalks_in_rig", []):
            points = _points_from_context(crosswalk, minimum_cols=2)
            if len(points) >= 3:
                crosswalks.append(points)

    return PDMPlannerInput(
        ego=PDMEgoState(
            speed_mps=max(0.0, float(speed_mps)),
            acceleration_mps2=float(acceleration_mps2),
            yaw_rad=ego_yaw,
        ),
        route_waypoints_in_rig=route_waypoints,
        nearby_lane_centerlines_in_rig=nearby_lane_centerlines,
        actors=actors,
        wait_lines_in_rig=wait_lines,
        crosswalks_in_rig=crosswalks,
        fallback_command=fallback_command,
    )


class PDMClosedLoopPlanner:
    """Dependency-free closed-loop planner with proposal generation and scoring."""

    def __init__(
        self,
        *,
        horizon_s: float,
        output_frequency_hz: int,
        min_turn_radius_m: float,
        max_accel_mps2: float,
        max_speed_mps: float,
    ) -> None:
        self._horizon_s = horizon_s
        self._output_frequency_hz = output_frequency_hz
        self._num_waypoints = max(1, int(round(horizon_s * output_frequency_hz)))
        self._dt = 1.0 / output_frequency_hz
        self._min_turn_radius_m = min_turn_radius_m
        self._max_accel_mps2 = max_accel_mps2
        self._max_speed_mps = max_speed_mps

    def plan(self, planner_input: PDMPlannerInput) -> PDMClosedLoopResult:
        start = perf_counter()
        route_available = len(planner_input.route_waypoints_in_rig) >= 2
        nearby_lane_count = len(planner_input.nearby_lane_centerlines_in_rig)

        proposals = self._build_proposals(planner_input)
        fallback_reason: str | None = None
        if not proposals:
            proposals = [
                self._fallback_centerline(
                    planner_input.fallback_command,
                    planner_input.ego.speed_mps,
                )
            ]
            fallback_reason = "heuristic_centerline"
        elif not route_available:
            fallback_reason = "missing_route"
        elif nearby_lane_count == 0:
            fallback_reason = "missing_nearby_lanes"

        scored_candidates = [
            self._simulate_and_score(planner_input, proposal_idx=i, source=source, centerline_xy=proposal)
            for i, (source, proposal) in enumerate(proposals)
        ]
        best = min(scored_candidates, key=lambda candidate: candidate.score)
        runtime_ms = (perf_counter() - start) * 1000.0

        return PDMClosedLoopResult(
            trajectory_xy=best.trajectory_xy.astype(np.float32),
            headings=best.headings.astype(np.float32),
            debug_metadata={
                "proposal_count": len(scored_candidates),
                "selected_proposal_idx": best.proposal_idx,
                "selected_proposal_source": best.source,
                "route_available": route_available,
                "nearby_lane_count": nearby_lane_count,
                "actor_count": len(planner_input.actors),
                "wait_line_count": len(planner_input.wait_lines_in_rig),
                "crosswalk_count": len(planner_input.crosswalks_in_rig),
                "fallback_reason": fallback_reason,
                "planner_runtime_ms": runtime_ms,
                "selected_score": best.score,
                "selected_score_breakdown": best.score_breakdown,
            },
        )

    def _build_proposals(
        self, planner_input: PDMPlannerInput
    ) -> list[tuple[str, np.ndarray]]:
        proposals: list[tuple[str, np.ndarray]] = []
        if len(planner_input.route_waypoints_in_rig) >= 2:
            proposals.append(("route_centerline", planner_input.route_waypoints_in_rig))
        for idx, lane in enumerate(planner_input.nearby_lane_centerlines_in_rig[:6]):
            proposals.append((f"nearby_lane_{idx}", lane))
        return proposals

    def _fallback_centerline(
        self, command: DriveCommand, speed_mps: float
    ) -> tuple[str, np.ndarray]:
        horizon_distance = max(speed_mps * self._horizon_s, 12.0)
        samples = np.linspace(0.0, horizon_distance, num=max(self._num_waypoints, 6))
        if command == DriveCommand.LEFT:
            radius = self._min_turn_radius_m
            theta = samples / radius
            centerline = np.column_stack(
                (radius * np.sin(theta), radius * (1.0 - np.cos(theta)))
            )
        elif command == DriveCommand.RIGHT:
            radius = self._min_turn_radius_m
            theta = samples / radius
            centerline = np.column_stack(
                (radius * np.sin(theta), -radius * (1.0 - np.cos(theta)))
            )
        else:
            centerline = np.column_stack((samples, np.zeros_like(samples)))
        return ("heuristic_fallback", centerline.astype(np.float64))

    def _simulate_and_score(
        self,
        planner_input: PDMPlannerInput,
        *,
        proposal_idx: int,
        source: str,
        centerline_xy: np.ndarray,
    ) -> _ProposalCandidate:
        trajectory_xy = _sample_along_polyline(
            centerline_xy,
            _motion_profile_distances(
                speed_mps=planner_input.ego.speed_mps,
                acceleration_mps2=planner_input.ego.acceleration_mps2,
                dt=self._dt,
                num_waypoints=self._num_waypoints,
                max_accel_mps2=self._max_accel_mps2,
                max_speed_mps=self._max_speed_mps,
            ),
        )
        headings = _compute_headings(trajectory_xy)
        score_breakdown = self._score_trajectory(
            planner_input=planner_input,
            centerline_xy=centerline_xy,
            trajectory_xy=trajectory_xy,
        )
        score = float(sum(score_breakdown.values()))
        return _ProposalCandidate(
            proposal_idx=proposal_idx,
            source=source,
            centerline_xy=centerline_xy,
            trajectory_xy=trajectory_xy,
            headings=headings,
            score=score,
            score_breakdown=score_breakdown,
        )

    def _score_trajectory(
        self,
        *,
        planner_input: PDMPlannerInput,
        centerline_xy: np.ndarray,
        trajectory_xy: np.ndarray,
    ) -> dict[str, float]:
        score_breakdown: dict[str, float] = {
            "progress_reward": -float(np.max(trajectory_xy[:, 0])),
            "curvature_penalty": _curvature_penalty(trajectory_xy),
            "actor_penalty": _actor_penalty(trajectory_xy, planner_input.actors),
            "lane_penalty": _lane_penalty(trajectory_xy, centerline_xy),
            "wait_line_penalty": _wait_line_penalty(
                trajectory_xy, planner_input.wait_lines_in_rig
            ),
            "crosswalk_penalty": _crosswalk_penalty(
                trajectory_xy, planner_input.crosswalks_in_rig
            ),
        }
        return score_breakdown


def _points_from_context(
    raw_points: Any,
    *,
    minimum_cols: int,
) -> np.ndarray:
    if not isinstance(raw_points, list) or len(raw_points) == 0:
        return np.zeros((0, minimum_cols), dtype=np.float64)
    try:
        points = np.asarray(raw_points, dtype=np.float64)
    except (TypeError, ValueError):
        return np.zeros((0, minimum_cols), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < minimum_cols:
        return np.zeros((0, minimum_cols), dtype=np.float64)
    return points[:, :minimum_cols]


def _motion_profile_distances(
    *,
    speed_mps: float,
    acceleration_mps2: float,
    dt: float,
    num_waypoints: int,
    max_accel_mps2: float,
    max_speed_mps: float,
) -> np.ndarray:
    t = np.arange(1, num_waypoints + 1, dtype=np.float64) * dt
    accel = float(np.clip(acceleration_mps2, -max_accel_mps2, max_accel_mps2))
    speed = float(np.clip(speed_mps, 0.0, max_speed_mps))
    return np.maximum(speed * t + 0.5 * accel * t**2, 0.0)


def _sample_along_polyline(polyline_xy: np.ndarray, distances: np.ndarray) -> np.ndarray:
    if len(polyline_xy) == 0:
        return np.column_stack((distances, np.zeros_like(distances)))
    if len(polyline_xy) == 1:
        return np.repeat(polyline_xy.astype(np.float64), len(distances), axis=0)

    deltas = np.diff(polyline_xy, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = cumulative[-1]
    if total_length <= 1e-6:
        return np.repeat(polyline_xy[:1].astype(np.float64), len(distances), axis=0)

    clipped = np.clip(distances, 0.0, total_length)
    out = np.zeros((len(clipped), 2), dtype=np.float64)
    for i, distance in enumerate(clipped):
        seg_idx = min(np.searchsorted(cumulative, distance, side="right") - 1, len(deltas) - 1)
        seg_len = max(segment_lengths[seg_idx], 1e-6)
        alpha = (distance - cumulative[seg_idx]) / seg_len
        out[i] = polyline_xy[seg_idx] + alpha * deltas[seg_idx]
    return out


def _compute_headings(trajectory_xy: np.ndarray) -> np.ndarray:
    prev = np.zeros_like(trajectory_xy)
    prev[1:, :] = trajectory_xy[:-1, :]
    deltas = trajectory_xy - prev
    return np.arctan2(deltas[:, 1], deltas[:, 0])


def _curvature_penalty(trajectory_xy: np.ndarray) -> float:
    if len(trajectory_xy) < 3:
        return 0.0
    headings = _compute_headings(trajectory_xy)
    heading_deltas = np.diff(headings)
    return float(np.sum(np.abs(np.unwrap(heading_deltas))) * 0.4)


def _lane_penalty(trajectory_xy: np.ndarray, centerline_xy: np.ndarray) -> float:
    if len(centerline_xy) < 2:
        return 0.0
    sampled_centerline = _sample_along_polyline(
        centerline_xy,
        _trajectory_arc_lengths(trajectory_xy),
    )
    deviation = np.linalg.norm(trajectory_xy - sampled_centerline, axis=1)
    return float(np.mean(deviation) * 0.5)


def _actor_penalty(trajectory_xy: np.ndarray, actors: Sequence[PDMActor]) -> float:
    if not actors:
        return 0.0
    penalty = 0.0
    for actor in actors:
        deltas = trajectory_xy - actor.position_in_rig[np.newaxis, :]
        dists = np.linalg.norm(deltas, axis=1)
        min_dist = float(np.min(dists))
        if min_dist < 1.5:
            penalty += 1_000.0
        elif min_dist < 4.0:
            penalty += (4.0 - min_dist) * 15.0
    return penalty


def _wait_line_penalty(
    trajectory_xy: np.ndarray,
    wait_lines: Sequence[PDMWaitLine],
) -> float:
    if not wait_lines or len(trajectory_xy) == 0:
        return 0.0
    penalty = 0.0
    for wait_line in wait_lines:
        mean_x = float(np.mean(wait_line.points_in_rig[:, 0]))
        if mean_x <= 0.0:
            continue
        crossed = np.any(trajectory_xy[:, 0] >= mean_x)
        if crossed and mean_x < 8.0:
            penalty += 120.0
        elif crossed:
            penalty += 15.0
    return penalty


def _crosswalk_penalty(
    trajectory_xy: np.ndarray,
    crosswalks: Sequence[np.ndarray],
) -> float:
    if not crosswalks or len(trajectory_xy) == 0:
        return 0.0
    penalty = 0.0
    for polygon in crosswalks:
        min_xy = np.min(polygon[:, :2], axis=0)
        max_xy = np.max(polygon[:, :2], axis=0)
        inside = np.logical_and(
            np.logical_and(trajectory_xy[:, 0] >= min_xy[0], trajectory_xy[:, 0] <= max_xy[0]),
            np.logical_and(trajectory_xy[:, 1] >= min_xy[1], trajectory_xy[:, 1] <= max_xy[1]),
        )
        if np.any(inside):
            penalty += 20.0
    return penalty


def _trajectory_arc_lengths(trajectory_xy: np.ndarray) -> np.ndarray:
    if len(trajectory_xy) == 0:
        return np.zeros(0, dtype=np.float64)
    deltas = np.diff(trajectory_xy, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))
