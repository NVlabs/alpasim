# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Policy event for event-based simulation loop.

Opens the per-step StepContext, gathers observations (egopose, route,
recording ground truth), queries the driver, and writes the transformed
trajectory into the context for downstream pipeline events.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Optional

import numpy as np
from alpasim_runtime.decision import (
    DecisionBundle,
    DecisionSnapshot,
    DriverServiceBackendAdapter,
    SingleBackendDriverOrchestrator,
    build_input_snapshot_id,
)
from alpasim_runtime.events.base import EventPriority, EventQueue, RecurringEvent
from alpasim_runtime.events.state import RolloutState, ServiceBundle
from alpasim_runtime.observation_cache import ObservationFrame
from alpasim_runtime.route_generator import RouteGenerator
from alpasim_utils import geometry

logger = logging.getLogger(__name__)
try:
    from trajdata.maps.vec_map_elements import MapElementType
except Exception:  # pragma: no cover - optional dependency boundary
    MapElementType = None


def _pose_to_position_and_yaw(pose: geometry.Pose) -> tuple[list[float], float]:
    """Extract position xyz and yaw from a Pose."""
    position, quat_wxyz = pose.to_proto()
    w, x, y, z = quat_wxyz
    yaw = float(
        np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
    )
    return [float(position[0]), float(position[1]), float(position[2])], yaw


def _local_to_rig_xy(
    point_xy: np.ndarray,
    ego_xy: np.ndarray,
    ego_yaw: float,
) -> np.ndarray:
    """Transform a 2D local-frame point into the current rig frame."""
    dxdy = np.asarray(point_xy[:2], dtype=np.float64) - ego_xy
    cos_yaw = float(np.cos(ego_yaw))
    sin_yaw = float(np.sin(ego_yaw))
    return np.array(
        [
            cos_yaw * dxdy[0] + sin_yaw * dxdy[1],
            -sin_yaw * dxdy[0] + cos_yaw * dxdy[1],
        ],
        dtype=np.float64,
    )


def _polyline_points_xy(obj: Any) -> np.ndarray | None:
    """Extract [N,2] points from map polyline/polygon-like objects."""
    points = getattr(obj, "points", None)
    if points is None:
        return None
    points_np = np.asarray(points)
    if points_np.ndim != 2 or points_np.shape[0] == 0 or points_np.shape[1] < 2:
        return None
    return points_np[:, :2].astype(np.float64)


def _build_planner_context(
    state: RolloutState,
    timestamp_us: int,
    route_in_rig: Optional[geometry.Polyline],
    max_actors: int = 64,
) -> dict[str, Any]:
    """Build a compact per-frame planning context for the driver."""
    ego_pose = state.ego_trajectory_estimate.last_pose
    ego_position, ego_yaw = _pose_to_position_and_yaw(ego_pose)
    ego_xy = np.array(ego_position[:2], dtype=np.float64)

    actor_rows: list[tuple[float, dict[str, Any]]] = []
    sample_ts = np.array([timestamp_us], dtype=np.uint64)
    for actor_id, actor in state.traffic_objs.items():
        if actor.is_static:
            continue
        if timestamp_us not in actor.trajectory.time_range_us:
            continue
        actor_pose = actor.trajectory.interpolate_poses_list(sample_ts)[0]
        actor_pos, actor_yaw = _pose_to_position_and_yaw(actor_pose)
        actor_xy = np.array(actor_pos[:2], dtype=np.float64)
        actor_rig_xy = _local_to_rig_xy(actor_xy, ego_xy, ego_yaw)
        distance = float(np.linalg.norm(actor_xy - ego_xy))
        actor_rows.append(
            (
                distance,
                {
                    "id": actor_id,
                    "label": actor.label_class,
                    "position_in_local": actor_pos,
                    "position_in_rig": [float(actor_rig_xy[0]), float(actor_rig_xy[1])],
                    "yaw": actor_yaw,
                    "distance": distance,
                },
            )
        )

    actor_rows.sort(key=lambda item: item[0])
    actors = [row for _, row in actor_rows[:max_actors]]

    route_waypoints: list[list[float]] = []
    if route_in_rig is not None:
        waypoints_np = np.asarray(route_in_rig.waypoints)
        if waypoints_np.size > 0:
            route_waypoints = waypoints_np[:, :3].astype(float).tolist()

    vector_map = state.unbound.vector_map
    map_summary: dict[str, Any] = {"available": vector_map is not None}
    nearby_lanes: list[dict[str, Any]] = []
    traffic_rules: dict[str, Any] = {
        "traffic_sign_count": 0,
        "wait_lines_in_rig": [],
        "crosswalks_in_rig": [],
    }
    if vector_map is not None:
        map_summary["map_id"] = getattr(vector_map, "map_id", None)
        lanes = getattr(vector_map, "lanes", None)
        road_edges = getattr(vector_map, "road_edges", None)
        map_summary["lane_count"] = len(lanes) if lanes is not None else None
        map_summary["road_edge_count"] = (
            len(road_edges) if road_edges is not None else None
        )

        if lanes is not None:
            lane_candidates: list[tuple[float, dict[str, Any]]] = []
            for lane in lanes:
                lane_id = getattr(lane, "id", None)
                center = getattr(lane, "center", None)
                center_xy = _polyline_points_xy(center) if center is not None else None
                if center_xy is None:
                    continue

                lane_center_xy = center_xy.mean(axis=0)
                lane_dist = float(np.linalg.norm(lane_center_xy - ego_xy))
                center_in_rig = np.array(
                    [_local_to_rig_xy(p, ego_xy, ego_yaw) for p in center_xy],
                    dtype=np.float64,
                )
                lane_candidates.append(
                    (
                        lane_dist,
                        {
                            "id": lane_id,
                            "centerline_in_rig": center_in_rig.tolist(),
                        },
                    )
                )

            lane_candidates.sort(key=lambda item: item[0])
            nearby_lanes = [lane_info for _, lane_info in lane_candidates[:20]]

        if MapElementType is not None:
            elements = getattr(vector_map, "elements", {}) or {}
            traffic_signs = elements.get(MapElementType.TRAFFIC_SIGN, {})
            wait_lines = elements.get(MapElementType.WAIT_LINE, {})
            crosswalks = elements.get(MapElementType.PED_CROSSWALK, {})
            traffic_rules["traffic_sign_count"] = len(traffic_signs)

            for wait_line in wait_lines.values():
                wl_type = str(getattr(wait_line, "wait_line_type", "Stop"))
                polyline = getattr(wait_line, "polyline", None)
                points_xy = _polyline_points_xy(polyline) if polyline is not None else None
                if points_xy is None:
                    continue
                points_in_rig = np.array(
                    [_local_to_rig_xy(p, ego_xy, ego_yaw) for p in points_xy],
                    dtype=np.float64,
                )
                traffic_rules["wait_lines_in_rig"].append(
                    {
                        "type": wl_type,
                        "points": points_in_rig.tolist(),
                    }
                )

            for crosswalk in crosswalks.values():
                polygon = getattr(crosswalk, "polygon", None)
                points_xy = _polyline_points_xy(polygon) if polygon is not None else None
                if points_xy is None:
                    continue
                points_in_rig = np.array(
                    [_local_to_rig_xy(p, ego_xy, ego_yaw) for p in points_xy],
                    dtype=np.float64,
                )
                traffic_rules["crosswalks_in_rig"].append(points_in_rig.tolist())

    return {
        "timestamp_us": int(timestamp_us),
        "ego": {
            "position": ego_position,
            "yaw": ego_yaw,
        },
        "route_waypoints_in_rig": route_waypoints,
        "actors": actors,
        "nearby_lanes": nearby_lanes,
        "traffic_rules": traffic_rules,
        "map_summary": map_summary,
    }


def _compute_policy_step_id(state: RolloutState, timestamp_us: int, interval_us: int) -> int:
    """Compute a deterministic per-step id for policy evaluation."""
    scene_start_us = int(state.unbound.control_timestamps_us[0])
    if interval_us <= 0:
        raise ValueError(f"policy interval must be positive, got {interval_us}")
    return int((timestamp_us - scene_start_us) // interval_us)


def _build_decision_snapshot(
    state: RolloutState,
    *,
    step_id: int,
    time_now_us: int,
    time_query_us: int,
    planner_context: dict[str, Any] | None,
    route_in_rig: Optional[geometry.Polyline],
    renderer_data: bytes | None,
) -> DecisionSnapshot:
    """Build a stable snapshot identity for the current policy input."""
    route_waypoints_in_rig: list[list[float]] = []
    if route_in_rig is not None:
        route_waypoints_np = np.asarray(route_in_rig.waypoints)
        if route_waypoints_np.size > 0:
            route_waypoints_in_rig = route_waypoints_np[:, :3].astype(float).tolist()

    camera_frame_timestamps_us = dict(sorted(state.last_camera_frame_us.items()))
    ego_pose_history_timestamps_us = [
        int(timestamp) for timestamp in state.ego_trajectory_estimate.timestamps_us.tolist()
    ]
    traffic_actor_ids = sorted(state.traffic_objs.keys())
    input_snapshot_id = build_input_snapshot_id(
        step_id=step_id,
        time_now_us=time_now_us,
        time_query_us=time_query_us,
        planner_context=planner_context,
        route_waypoints_in_rig=route_waypoints_in_rig,
        traffic_actor_ids=traffic_actor_ids,
        ego_pose_history_timestamps_us=ego_pose_history_timestamps_us,
        camera_frame_timestamps_us=camera_frame_timestamps_us,
        renderer_data=renderer_data,
    )
    return DecisionSnapshot(
        step_id=step_id,
        input_snapshot_id=input_snapshot_id,
        time_now_us=time_now_us,
        time_query_us=time_query_us,
        ego_pose_history_timestamps_us=ego_pose_history_timestamps_us,
        traffic_actor_ids=traffic_actor_ids,
        route_waypoints_in_rig=route_waypoints_in_rig,
        planner_context=planner_context,
        renderer_data=renderer_data,
        camera_frame_timestamps_us=camera_frame_timestamps_us,
    )


def _append_observation_frame(
    state: RolloutState,
    *,
    decision_snapshot: DecisionSnapshot,
    planner_context: dict[str, Any] | None,
    route_in_rig: geometry.Polyline | None,
    renderer_data: bytes | None,
) -> None:
    if state.observation_cache is None:
        return
    state.observation_cache.append(
        ObservationFrame(
            step_id=decision_snapshot.step_id,
            input_snapshot_id=decision_snapshot.input_snapshot_id,
            time_now_us=decision_snapshot.time_now_us,
            time_query_us=decision_snapshot.time_query_us,
            camera_frame_timestamps_us=dict(decision_snapshot.camera_frame_timestamps_us),
            rendered_images=dict(state.last_rendered_images),
            renderer_data=renderer_data,
            ego_trajectory=state.ego_trajectory.clone(),
            ego_trajectory_estimate=state.ego_trajectory_estimate.clone(),
            traffic_objs=state.traffic_objs.clip_trajectories(
                int(state.unbound.control_timestamps_us[0]),
                int(state.unbound.control_timestamps_us[-1]) + 1,
            ),
            ego_pose_history_timestamps_us=list(
                decision_snapshot.ego_pose_history_timestamps_us
            ),
            route_waypoints_in_rig=(
                []
                if route_in_rig is None
                else [list(point) for point in route_in_rig.points.tolist()]
            ),
            planner_context=planner_context,
            active_backend_ids=(
                list(state.active_driver_backend_ids)
                if state.active_driver_backend_ids is not None
                else list(state.available_driver_backend_ids)
            ),
        )
    )


class PolicyEvent(RecurringEvent):
    """Open per-step context, gather observations, query driver.

    Handles egopose submission, sync tracking, route updates, and recording
    ground-truth submission.  Everything after the driver query (controller,
    physics, traffic, commit) is handled by downstream pipeline events.
    """

    priority: int = EventPriority.POLICY

    def __init__(
        self,
        timestamp_us: int,
        policy_timestep_us: int,
        services: ServiceBundle,
        camera_ids: list[str],
        route_generator: Optional[RouteGenerator],
        send_recording_ground_truth: bool,
    ):
        super().__init__(timestamp_us=timestamp_us)
        self.interval_us = policy_timestep_us
        self.services = services
        self.camera_ids = camera_ids
        self.route_generator = route_generator
        self.send_recording_ground_truth = send_recording_ground_truth

    async def run(self, state: RolloutState, queue: EventQueue) -> None:
        step_start_us = self.timestamp_us
        target_time_us = step_start_us + self.interval_us
        svc = self.services

        # --- Step boundary: fill timing on existing StepContext ---
        assert (
            state.step_context is not None
        ), "StepContext must exist before PolicyEvent (created by StepEvent)"
        state.step_context.step_start_us = step_start_us
        state.step_context.target_time_us = target_time_us
        state.step_context.force_gt = target_time_us in state.unbound.force_gt_period

        # --- Sensor sync validation ---
        if state.unbound.assert_zero_decision_delay:
            assert_sensors_up_to_date(state, step_start_us, self.camera_ids)

        # --- Submit observations concurrently ---
        # Send all egomotion observations since the last update, not just the
        # latest one.  When pose_reporting_interval_us > 0 the controller
        # produces intermediate poses that StepEvent appends to the estimated
        # trajectory.  The driver should receive every one of them.
        all_ts = state.ego_trajectory_estimate.timestamps_us
        mask = (all_ts > state.last_egopose_update_us) & (all_ts <= step_start_us)
        ts_arr = all_ts[mask]
        if len(ts_arr) == 0:
            ts_arr = np.array([step_start_us], dtype=np.uint64)

        ego_trajectory = state.ego_trajectory_estimate.trajectory().interpolate(ts_arr)
        dynamics_arr = state.ego_trajectory_estimate.interpolate_dynamics(ts_arr)
        dynamic_states_in_rig = geometry.array_to_dynamic_states(dynamics_arr)

        if (self.route_generator is not None or self.send_recording_ground_truth) and (
            state.ego_trajectory.timestamps_us[-1] != step_start_us
        ):
            raise ValueError(
                f"Timestamp mismatch: {state.ego_trajectory.timestamps_us[-1]} "
                f"!= {step_start_us}"
            )

        ctx = state.step_context
        ctx.track_task(
            svc.driver.submit_trajectory(ego_trajectory, dynamic_states_in_rig)
        )

        route_for_policy: Optional[geometry.Polyline] = None
        if self.route_generator is not None:
            pose_local_to_rig = state.ego_trajectory.last_pose
            route = self.route_generator.generate_route(
                step_start_us, pose_local_to_rig
            )
            route_for_policy = RouteGenerator.prepare_for_policy(route)
            ctx.track_task(svc.driver.submit_route(step_start_us, route_for_policy))

        if self.send_recording_ground_truth:
            gt_traj = state.unbound.gt_ego_trajectory
            pose_local_to_rig = state.ego_trajectory.last_pose
            traj_in_rig = gt_traj.transform(pose_local_to_rig.inverse())
            ctx.track_task(
                svc.driver.submit_recording_ground_truth(step_start_us, traj_in_rig)
            )

        # Barrier: all observations (images + egopose + route + GT) must
        # reach the driver before we call drive().
        await ctx.drain_outstanding_tasks()
        state.last_egopose_update_us = step_start_us

        planner_context = _build_planner_context(
            state=state,
            timestamp_us=step_start_us,
            route_in_rig=route_for_policy,
        )
        step_id = _compute_policy_step_id(
            state,
            timestamp_us=step_start_us,
            interval_us=self.interval_us,
        )
        decision_snapshot = _build_decision_snapshot(
            state,
            step_id=step_id,
            time_now_us=step_start_us,
            time_query_us=target_time_us,
            planner_context=planner_context,
            route_in_rig=route_for_policy,
            renderer_data=state.data_sensorsim_to_driver,
        )
        _append_observation_frame(
            state,
            decision_snapshot=decision_snapshot,
            planner_context=planner_context,
            route_in_rig=route_for_policy,
            renderer_data=state.data_sensorsim_to_driver,
        )
        orchestrator = svc.driver_orchestrator or SingleBackendDriverOrchestrator(
            DriverServiceBackendAdapter(svc.driver)
        )
        decision_bundle = await orchestrator.generate_candidates(
            decision_snapshot,
            backend_ids=state.active_driver_backend_ids,
        )
        selected_candidate = await orchestrator.select_candidate(decision_bundle)
        state.data_sensorsim_to_driver = None  # Consumed

        transformed_candidates = []
        for candidate in decision_bundle.candidates:
            transformed_trajectory = candidate.trajectory
            if transformed_trajectory is not None:
                transformed_trajectory = transform_trajectory_from_noisy_to_true_local_frame(
                    state,
                    transformed_trajectory,
                )
            transformed_candidates.append(
                replace(candidate, trajectory=transformed_trajectory)
            )

        transformed_bundle = DecisionBundle(
            snapshot=decision_bundle.snapshot,
            candidates=transformed_candidates,
            selected_candidate_id=selected_candidate.candidate_id,
            arbitration_reason=decision_bundle.arbitration_reason or "single_backend_default",
        )
        selected_candidate = next(
            candidate
            for candidate in transformed_bundle.candidates
            if candidate.candidate_id == selected_candidate.candidate_id
        )
        state.last_decision_step_id = step_id
        state.step_context.decision_bundle = transformed_bundle
        state.step_context.driver_trajectory = selected_candidate.trajectory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_sensors_up_to_date(
    state: RolloutState, step_start_us: int, camera_ids: list[str]
) -> None:
    """Validate that egopose and all camera frames are current before the policy decision."""
    # --- egopose freshness ---
    latest_ego_us = int(state.ego_trajectory_estimate.timestamps_us[-1])
    if latest_ego_us != step_start_us:
        raise ValueError(
            f"Egopose not up to date at {step_start_us}: "
            f"ego_trajectory_estimate latest timestamp is {latest_ego_us}"
        )

    # --- camera freshness ---
    if not state.last_camera_frame_us:
        return  # First step — no cameras have fired yet

    stale = [
        cid
        for cid in camera_ids
        if state.last_camera_frame_us.get(cid, 0) != step_start_us
    ]
    if stale:
        raise ValueError(f"Cameras not up to date at {step_start_us}: {stale}")


def transform_trajectory_from_noisy_to_true_local_frame(
    state: RolloutState, drive_trajectory_noisy: geometry.Trajectory
) -> geometry.Trajectory:
    """Transform trajectory from noisy local frame to true local frame.

    The driver operates in the "noisy" (estimated) rig frame. To map its output
    into the true local frame we:

    1. Undo the estimated rig frame:  ``T_estimate_inv * traj``
    2. Apply the true rig frame:      ``T_true * result``

    When no egomotion noise model is active, ``ego_trajectory_estimate`` tracks
    ``ego_trajectory`` exactly and the transform is identity.  When noise is
    present the two trajectories diverge and this mapping compensates for the
    drift the driver doesn't know about.
    """
    return drive_trajectory_noisy.transform(
        state.ego_trajectory_estimate.last_pose.inverse()
    ).transform(state.ego_trajectory.last_pose)
