# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Build causal vector observations in the physical ego frame."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from alpasim_grpc.v0.common_pb2 import AABB, Vec3
from alpasim_grpc.v0.egodriver_pb2 import VectorizedObservation
from alpasim_utils.artifact import Artifact
from alpasim_utils.geometry import Polyline, Pose, Trajectory, trajectory_to_grpc
from alpasim_utils.scenario import TrafficObjects
from trajdata.maps import VectorMap
from trajdata.maps.vec_map_elements import MapElementType

_MAP_RANGE_M = 200.0
_DEFAULT_OBSTACLE_HISTORY_STEPS = 16
_DEFAULT_EGO_LENGTH_M = 4.8
_DEFAULT_EGO_WIDTH_M = 2.0


@dataclass(frozen=True)
class EgoFootprint:
    """Physical ego-frame rectangle used only by the sanity-check plot."""

    length_m: float
    width_m: float
    center_x_m: float
    center_y_m: float


def build_vectorized_observation(
    *,
    timestamp_us: int,
    vector_map: VectorMap | None,
    traffic_objects: TrafficObjects,
    pose_local_to_physical_ego: Pose,
    obstacle_history_steps: int = _DEFAULT_OBSTACLE_HISTORY_STEPS,
) -> VectorizedObservation:
    """Return map and causal obstacle histories in the physical ego frame.

    ``pose_local_to_physical_ego`` is the current physical ego pose. Map
    elements and traffic trajectories arrive in AlpaSim's local frame, so its
    inverse is applied before serialization. Traffic samples strictly after
    ``timestamp_us`` are excluded even if the backing scene contains them,
    and each actor contributes only a fixed trailing history.
    """
    if obstacle_history_steps <= 0:
        raise ValueError(
            f"obstacle_history_steps must be positive, got {obstacle_history_steps}"
        )

    local_to_ego = pose_local_to_physical_ego.inverse()
    observation = VectorizedObservation(timestamp_us=timestamp_us)
    if vector_map is not None:
        observation.lane_lines.extend(
            _map_polylines_in_ego(vector_map, MapElementType.ROAD_LANE, local_to_ego)
        )
        observation.road_boundaries.extend(
            _map_polylines_in_ego(vector_map, MapElementType.ROAD_EDGE, local_to_ego)
        )

    for track_id, traffic_object in sorted(traffic_objects.items()):
        trajectory = traffic_object.trajectory
        if trajectory.is_empty() or trajectory.time_range_us.start > timestamp_us:
            continue
        causal = trajectory.clip(trajectory.time_range_us.start, timestamp_us + 1)
        if causal.is_empty():
            continue
        if len(causal.timestamps_us) > obstacle_history_steps:
            causal = causal.slice(
                len(causal.timestamps_us) - obstacle_history_steps,
                len(causal.timestamps_us),
            )
        causal_in_ego = causal.transform(local_to_ego)
        in_range_indices = np.flatnonzero(_points_in_range(causal_in_ego.positions))
        if len(in_range_indices) == 0:
            continue
        obstacle = observation.obstacles.add(
            object_id=track_id,
            is_static=traffic_object.is_static,
            label_class=traffic_object.label_class,
        )
        obstacle.aabb.CopyFrom(
            AABB(
                size_x=float(traffic_object.aabb.x),
                size_y=float(traffic_object.aabb.y),
                size_z=float(traffic_object.aabb.z),
            )
        )
        obstacle.trajectory.CopyFrom(
            trajectory_to_grpc(
                Trajectory.from_poses(
                    causal_in_ego.timestamps_us[in_range_indices],
                    [causal_in_ego.get_pose(int(index)) for index in in_range_indices],
                )
            )
        )
    return observation


def _map_polylines_in_ego(
    vector_map: VectorMap,
    element_type: MapElementType,
    local_to_ego: Pose,
) -> list[VectorizedObservation.Polyline]:
    """Extract one map layer, transform it, and keep nearby polylines."""
    ego_xyz_local = local_to_ego.inverse().vec3
    raw_polylines: list[np.ndarray] = []
    for element in _nearby_map_elements(vector_map, element_type, ego_xyz_local):
        if element_type == MapElementType.ROAD_LANE:
            for edge in (element.left_edge, element.right_edge):
                if edge is not None:
                    raw_polylines.append(edge.xyz)
        else:
            raw_polylines.append(element.polyline.xyz)

    result: list[VectorizedObservation.Polyline] = []
    for raw in raw_polylines:
        points = np.asarray(raw, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
            continue
        if not _points_hit_local_range(points, ego_xyz_local[:2]):
            continue
        if points.shape[1] == 2:
            points = np.column_stack(
                [points, np.zeros((points.shape[0],), dtype=np.float32)]
            )
        points_in_ego = Polyline(points=points[:, :3]).transform(local_to_ego).waypoints
        if not _points_hit_range(points_in_ego):
            continue
        for clipped_points in _clip_polyline_to_range(points_in_ego):
            result.append(
                VectorizedObservation.Polyline(
                    points=[
                        Vec3(x=float(x), y=float(y), z=float(z))
                        for x, y, z in clipped_points
                    ]
                )
            )
    return result


def _nearby_map_elements(
    vector_map: VectorMap,
    element_type: MapElementType,
    ego_xyz_local: np.ndarray,
) -> list[Any]:
    range_m = _MAP_RANGE_M * math.sqrt(2.0)
    try:
        if element_type == MapElementType.ROAD_LANE:
            return list(vector_map.get_lanes_within(ego_xyz_local, range_m))
        if element_type == MapElementType.ROAD_EDGE:
            return list(vector_map.get_road_edges_within(ego_xyz_local, range_m))
    except (AssertionError, AttributeError, KeyError, ValueError):
        pass
    return list(vector_map.elements.get(element_type, {}).values())


def _points_hit_local_range(points: np.ndarray, center_xy: np.ndarray) -> bool:
    """Cheap local-frame bounds check before ego-frame transforms."""
    if points.size == 0:
        return False
    xy = np.asarray(points)[:, :2]
    range_m = _MAP_RANGE_M * math.sqrt(2.0)
    return bool(
        xy[:, 0].min() <= center_xy[0] + range_m
        and xy[:, 0].max() >= center_xy[0] - range_m
        and xy[:, 1].min() <= center_xy[1] + range_m
        and xy[:, 1].max() >= center_xy[1] - range_m
    )


def _points_in_range(points: np.ndarray, range_m: float = _MAP_RANGE_M) -> np.ndarray:
    """Return a mask for points inside the configured ego box."""
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    xy = np.asarray(points)[:, :2]
    return (
        (xy[:, 0] >= -range_m)
        & (xy[:, 0] <= range_m)
        & (xy[:, 1] >= -range_m)
        & (xy[:, 1] <= range_m)
    )


def _clip_polyline_to_range(points: np.ndarray) -> list[np.ndarray]:
    """Clip a 3D polyline to the configured ego XY box."""
    clipped: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for start, end in pairwise(points):
        segment = _clip_segment_to_range(start, end)
        if segment is None:
            if current:
                clipped.append(np.asarray(current, dtype=np.float32))
                current = []
            continue
        clipped_start, clipped_end = segment
        if np.allclose(clipped_start, clipped_end):
            continue
        if current and np.allclose(current[-1], clipped_start):
            current.append(clipped_end)
        else:
            if current:
                clipped.append(np.asarray(current, dtype=np.float32))
            current = [clipped_start, clipped_end]
    if current:
        clipped.append(np.asarray(current, dtype=np.float32))
    return [part for part in clipped if len(part) >= 2]


def _clip_segment_to_range(
    start: np.ndarray, end: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return segment endpoints clipped to the configured ego XY box."""
    delta = end - start
    t_min = 0.0
    t_max = 1.0
    bounds = (
        (-delta[0], start[0] + _MAP_RANGE_M),
        (delta[0], _MAP_RANGE_M - start[0]),
        (-delta[1], start[1] + _MAP_RANGE_M),
        (delta[1], _MAP_RANGE_M - start[1]),
    )
    for edge_normal_dot_delta, edge_distance in bounds:
        if abs(edge_normal_dot_delta) < 1e-12:
            if edge_distance < 0.0:
                return None
            continue
        t = edge_distance / edge_normal_dot_delta
        if edge_normal_dot_delta < 0.0:
            if t > t_max:
                return None
            t_min = max(t_min, float(t))
        else:
            if t < t_min:
                return None
            t_max = min(t_max, float(t))
    clipped_start = start + t_min * delta
    clipped_end = start + t_max * delta
    return clipped_start.astype(np.float32), clipped_end.astype(np.float32)


def _points_hit_range(points: np.ndarray) -> bool:
    """Return whether a point set's XY bounds intersect the configured ego box."""
    if points.size == 0:
        return False
    xy = np.asarray(points)[:, :2]
    return bool(
        xy[:, 0].min() <= _MAP_RANGE_M
        and xy[:, 0].max() >= -_MAP_RANGE_M
        and xy[:, 1].min() <= _MAP_RANGE_M
        and xy[:, 1].max() >= -_MAP_RANGE_M
    )


def plot_vectorized_observation(
    observation: VectorizedObservation,
    *,
    ax: Any | None = None,
    ego_footprint: EgoFootprint | None = None,
    range_m: float = _MAP_RANGE_M,
) -> Any:
    """Plot the exact physical ego-frame vectors sent to an ego driver.

    Lane lines are blue, road boundaries are orange, obstacle trajectories are
    solid magenta lines, and each obstacle's latest oriented box is outlined in
    red. The ego vehicle is a rectangle in the current rig frame, whose origin
    remains marked at ``(0, 0)`` with +X forward and +Y left. Returns the
    Matplotlib axes so callers can add annotations or save it.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    if not math.isfinite(range_m) or range_m <= 0.0:
        raise ValueError(f"range_m must be finite and positive, got {range_m}")
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    for index, polyline in enumerate(observation.lane_lines):
        xy = np.array([(point.x, point.y) for point in polyline.points], dtype=float)
        if len(xy) >= 2:
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                color="tab:blue",
                linewidth=1.0,
                label="lane lines" if index == 0 else None,
            )
    for index, polyline in enumerate(observation.road_boundaries):
        xy = np.array([(point.x, point.y) for point in polyline.points], dtype=float)
        if len(xy) >= 2:
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                color="tab:orange",
                linewidth=1.5,
                label="road boundaries" if index == 0 else None,
            )

    for index, obstacle in enumerate(observation.obstacles):
        poses = obstacle.trajectory.poses
        xy = np.array(
            [(sample.pose.vec.x, sample.pose.vec.y) for sample in poses],
            dtype=float,
        )
        if len(xy) == 0:
            continue
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            "-",
            color="tab:purple",
            linewidth=1.0,
            label="obstacle trajectories" if index == 0 else None,
        )
        latest = poses[-1].pose
        yaw = _proto_pose_yaw(latest)
        half_length = 0.5 * float(obstacle.aabb.size_x)
        half_width = 0.5 * float(obstacle.aabb.size_y)
        local_corners = np.array(
            [
                [half_length, half_width],
                [half_length, -half_width],
                [-half_length, -half_width],
                [-half_length, half_width],
            ]
        )
        rotation = np.array(
            [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
        )
        corners = local_corners @ rotation.T + xy[-1]
        ax.add_patch(
            Polygon(
                corners,
                closed=True,
                fill=False,
                edgecolor="tab:red",
                linewidth=1.0,
            )
        )

    footprint = ego_footprint or EgoFootprint(
        length_m=_DEFAULT_EGO_LENGTH_M,
        width_m=_DEFAULT_EGO_WIDTH_M,
        center_x_m=0.0,
        center_y_m=0.0,
    )
    ax.add_patch(
        Rectangle(
            (
                footprint.center_x_m - 0.5 * footprint.length_m,
                footprint.center_y_m - 0.5 * footprint.width_m,
            ),
            footprint.length_m,
            footprint.width_m,
            facecolor="tab:green",
            edgecolor="black",
            linewidth=2.0,
            alpha=0.9,
            zorder=5,
            label="current ego",
        )
    )
    ax.scatter(
        [0.0], [0.0], marker="+", s=70, color="white", zorder=6, label="rig origin"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-range_m, range_m)
    ax.set_ylim(-range_m, range_m)
    ax.set_xlabel("forward x [m]")
    ax.set_ylabel("left y [m]")
    ax.set_title(f"Vectorized observation at {observation.timestamp_us} us")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    return ax


def save_vectorized_observation_plot(
    observation: VectorizedObservation,
    output_path: str | Path,
    *,
    ego_footprint: EgoFootprint | None = None,
    range_m: float = _MAP_RANGE_M,
) -> Path:
    """Render a vector observation to ``output_path`` and return the path."""
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ax = plot_vectorized_observation(
        observation,
        ego_footprint=ego_footprint,
        range_m=range_m,
    )
    ax.figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(ax.figure)
    return output


def vectorized_observation_from_usdz(
    usdz_path: str | Path,
    *,
    timestamp_us: int,
) -> VectorizedObservation:
    """Build the driver payload at ``timestamp_us`` directly from a USDZ scene."""
    artifact = Artifact(source=str(usdz_path))
    return _vectorized_observation_from_artifact(artifact, timestamp_us)


def _vectorized_observation_from_artifact(
    artifact: Artifact,
    timestamp_us: int,
) -> VectorizedObservation:
    """Build one vector payload from an already-loaded artifact."""
    ego_pose = artifact.rig.trajectory.interpolate(
        np.array([timestamp_us], dtype=np.uint64)
    ).last_pose
    return build_vectorized_observation(
        timestamp_us=timestamp_us,
        vector_map=artifact.map,
        traffic_objects=artifact.traffic_objects,
        pose_local_to_physical_ego=ego_pose,
    )


def sample_middle_timestamp_us(
    timestamps_us: np.ndarray,
    *,
    seed: int | None = None,
) -> int:
    """Randomly choose a recorded timestamp from the scene's middle half."""
    timestamps = np.asarray(timestamps_us, dtype=np.uint64).reshape(-1)
    if timestamps.size < 3:
        raise ValueError("middle timestamp sampling requires at least three ego poses")
    first_index = max(1, timestamps.size // 4)
    stop_index = min(timestamps.size - 1, (3 * timestamps.size + 3) // 4)
    candidates = timestamps[first_index:stop_index]
    if candidates.size == 0:
        raise ValueError("scene has no interior ego timestamp to sample")
    return int(np.random.default_rng(seed).choice(candidates))


def _ego_footprint_from_artifact(artifact: Artifact) -> EgoFootprint:
    """Return the USDZ ego AABB in current rig coordinates, or a visual default."""
    vehicle = artifact.rig.vehicle_config
    if vehicle is None:
        return EgoFootprint(
            length_m=_DEFAULT_EGO_LENGTH_M,
            width_m=_DEFAULT_EGO_WIDTH_M,
            center_x_m=0.0,
            center_y_m=0.0,
        )
    return EgoFootprint(
        length_m=float(vehicle.aabb_x_m),
        width_m=float(vehicle.aabb_y_m),
        center_x_m=float(vehicle.aabb_x_offset_m + vehicle.aabb_x_m / 2.0),
        center_y_m=float(vehicle.aabb_y_offset_m),
    )


def _proto_pose_yaw(pose: Any) -> float:
    """Return yaw from a protobuf pose's wxyz quaternion."""
    quat = pose.quat
    return math.atan2(
        2.0 * (quat.w * quat.z + quat.x * quat.y),
        1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z),
    )


def main() -> None:
    """Render one actual USDZ vector observation from the command line."""
    parser = argparse.ArgumentParser(
        description="Visualize the physical ego-frame vector payload sent to an ego driver."
    )
    parser.add_argument("usdz", type=Path)
    parser.add_argument(
        "--timestamp-us",
        type=int,
        help="Exact timestamp. Omit to sample randomly from the scene's middle half.",
    )
    parser.add_argument(
        "--seed", type=int, help="Optional seed for middle-timestamp sampling."
    )
    parser.add_argument(
        "--range-m",
        type=float,
        default=_MAP_RANGE_M,
        help="Symmetric XY plot range around current ego in meters.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = Artifact(source=str(args.usdz))
    timestamp_us = (
        args.timestamp_us
        if args.timestamp_us is not None
        else sample_middle_timestamp_us(
            artifact.rig.trajectory.timestamps_us, seed=args.seed
        )
    )
    observation = _vectorized_observation_from_artifact(artifact, timestamp_us)
    footprint = _ego_footprint_from_artifact(artifact)
    output = save_vectorized_observation_plot(
        observation,
        args.output,
        ego_footprint=footprint,
        range_m=args.range_m,
    )
    print(
        f"wrote {output}: timestamp_us={timestamp_us} "
        f"lane_lines={len(observation.lane_lines)} "
        f"road_boundaries={len(observation.road_boundaries)} "
        f"obstacles={len(observation.obstacles)}"
    )


if __name__ == "__main__":
    main()
