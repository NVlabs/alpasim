# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for current-driver-ego-frame vector observation construction."""

import numpy as np
from alpasim_runtime.vectorized_observation import (
    EgoFootprint,
    build_vectorized_observation,
    plot_vectorized_observation,
    sample_middle_timestamp_us,
    save_vectorized_observation_plot,
)
from alpasim_utils.geometry import Pose, Trajectory
from alpasim_utils.scenario import AABB, TrafficObject, TrafficObjects
from trajdata.maps import VectorMap
from trajdata.maps.vec_map_elements import Polyline, RoadEdge, RoadLane


def _pose(x: float, y: float = 0.0) -> Pose:
    """Return an identity-heading pose at ``(x, y)``."""
    return Pose(
        np.array([x, y, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )


def test_vectors_use_current_ego_frame_and_obstacles_are_causal() -> None:
    """Map and actors are ego-relative and no future actor pose is serialized."""
    vector_map = VectorMap(map_id="test:map")
    vector_map.add_map_element(
        RoadLane(
            id="lane",
            center=Polyline(np.array([[10.0, 0.0, 0.0], [12.0, 0.0, 0.0]])),
            left_edge=Polyline(np.array([[10.0, 1.0, 0.0], [12.0, 1.0, 0.0]])),
            right_edge=Polyline(np.array([[10.0, -1.0, 0.0], [12.0, -1.0, 0.0]])),
        )
    )
    vector_map.add_map_element(
        RoadEdge(
            id="edge",
            polyline=Polyline(np.array([[10.0, 3.0, 0.0], [12.0, 3.0, 0.0]])),
        )
    )
    obstacle_trajectory = Trajectory.from_poses(
        np.array([0, 100, 200], dtype=np.uint64),
        [_pose(12.0), _pose(13.0), _pose(99.0)],
    )
    traffic = TrafficObjects(
        actor=TrafficObject(
            track_id="actor",
            aabb=AABB(4.0, 2.0, 1.5),
            trajectory=obstacle_trajectory,
            is_static=False,
            label_class="vehicle",
        )
    )

    observation = build_vectorized_observation(
        timestamp_us=100,
        vector_map=vector_map,
        traffic_objects=traffic,
        pose_local_to_physical_ego=_pose(10.0),
    )

    assert observation.timestamp_us == 100
    assert len(observation.lane_lines) == 2
    assert observation.lane_lines[0].points[0].x == 0.0
    assert len(observation.road_boundaries) == 1
    assert observation.road_boundaries[0].points[0].y == 3.0
    assert len(observation.obstacles) == 1
    obstacle = observation.obstacles[0]
    assert obstacle.object_id == "actor"
    assert not obstacle.is_static
    assert obstacle.label_class == "vehicle"
    assert [pose.timestamp_us for pose in obstacle.trajectory.poses] == [0, 100]
    assert [pose.pose.vec.x for pose in obstacle.trajectory.poses] == [2.0, 3.0]


def test_obstacle_history_is_bounded_to_latest_samples() -> None:
    traffic = TrafficObjects(
        actor=TrafficObject(
            track_id="actor",
            aabb=AABB(4.0, 2.0, 1.5),
            trajectory=Trajectory.from_poses(
                np.array([0, 100, 200, 300, 400], dtype=np.uint64),
                [_pose(float(index)) for index in range(5)],
            ),
            is_static=False,
            label_class="vehicle",
        )
    )

    observation = build_vectorized_observation(
        timestamp_us=400,
        vector_map=None,
        traffic_objects=traffic,
        pose_local_to_physical_ego=_pose(0.0),
        obstacle_history_steps=3,
    )

    assert [
        pose.timestamp_us for pose in observation.obstacles[0].trajectory.poses
    ] == [
        200,
        300,
        400,
    ]


def test_far_vectors_are_not_transported() -> None:
    """Features outside the bounded current-ego window are omitted."""
    vector_map = VectorMap(map_id="test:map")
    vector_map.add_map_element(
        RoadEdge(
            id="far",
            polyline=Polyline(np.array([[500.0, 500.0, 0.0], [510.0, 500.0, 0.0]])),
        )
    )
    observation = build_vectorized_observation(
        timestamp_us=0,
        vector_map=vector_map,
        traffic_objects=TrafficObjects(),
        pose_local_to_physical_ego=_pose(0.0),
    )
    assert not observation.road_boundaries


def test_crossing_polylines_are_clipped_to_payload_range() -> None:
    vector_map = VectorMap(map_id="test:map")
    vector_map.add_map_element(
        RoadEdge(
            id="crossing",
            polyline=Polyline(
                np.array([[-500.0, 0.0, 0.0], [0.0, 0.0, 0.0], [500.0, 0.0, 0.0]])
            ),
        )
    )

    observation = build_vectorized_observation(
        timestamp_us=0,
        vector_map=vector_map,
        traffic_objects=TrafficObjects(),
        pose_local_to_physical_ego=_pose(0.0),
    )

    assert len(observation.road_boundaries) == 1
    points = observation.road_boundaries[0].points
    np.testing.assert_allclose([point.x for point in points], [-200.0, 0.0, 200.0])
    assert all(abs(point.x) <= 200.0 and abs(point.y) <= 200.0 for point in points)


def test_obstacle_histories_drop_out_of_range_samples() -> None:
    traffic = TrafficObjects(
        actor=TrafficObject(
            track_id="actor",
            aabb=AABB(4.0, 2.0, 1.5),
            trajectory=Trajectory.from_poses(
                np.array([0, 100, 200], dtype=np.uint64),
                [_pose(250.0), _pose(150.0), _pose(100.0)],
            ),
            is_static=False,
            label_class="vehicle",
        )
    )

    observation = build_vectorized_observation(
        timestamp_us=200,
        vector_map=None,
        traffic_objects=traffic,
        pose_local_to_physical_ego=_pose(0.0),
    )

    assert len(observation.obstacles) == 1
    poses = observation.obstacles[0].trajectory.poses
    assert [pose.timestamp_us for pose in poses] == [100, 200]
    np.testing.assert_allclose([pose.pose.vec.x for pose in poses], [150.0, 100.0])


def test_vectorized_observation_plot_is_written(tmp_path) -> None:
    """The reusable visualizer renders the exact protobuf payload."""
    observation = build_vectorized_observation(
        timestamp_us=0,
        vector_map=None,
        traffic_objects=TrafficObjects(),
        pose_local_to_physical_ego=_pose(0.0),
    )
    output = save_vectorized_observation_plot(observation, tmp_path / "vectors.png")
    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_draws_actual_ego_rectangle_and_rig_origin() -> None:
    """The sanity plot shows the ego AABB instead of a point marker."""
    observation = build_vectorized_observation(
        timestamp_us=100,
        vector_map=None,
        traffic_objects=TrafficObjects(),
        pose_local_to_physical_ego=_pose(0.0),
    )
    ax = plot_vectorized_observation(
        observation,
        ego_footprint=EgoFootprint(
            length_m=5.0,
            width_m=2.0,
            center_x_m=1.5,
            center_y_m=0.0,
        ),
        range_m=25.0,
    )

    ego_rectangle = ax.patches[0]
    assert np.isclose(ego_rectangle.get_x(), -1.0)
    assert np.isclose(ego_rectangle.get_y(), -1.0)
    assert np.isclose(ego_rectangle.get_width(), 5.0)
    assert np.isclose(ego_rectangle.get_height(), 2.0)
    assert ax.get_xlim() == (-25.0, 25.0)
    assert ax.get_ylim() == (-25.0, 25.0)


def test_plot_obstacle_trajectory_has_no_point_markers() -> None:
    """Obstacle trajectories render as clean lines without purple sample dots."""
    traffic = TrafficObjects(
        actor=TrafficObject(
            track_id="actor",
            aabb=AABB(4.0, 2.0, 1.5),
            trajectory=Trajectory.from_poses(
                np.array([0, 100], dtype=np.uint64),
                [_pose(1.0), _pose(2.0)],
            ),
            is_static=False,
            label_class="vehicle",
        )
    )
    observation = build_vectorized_observation(
        timestamp_us=100,
        vector_map=None,
        traffic_objects=traffic,
        pose_local_to_physical_ego=_pose(0.0),
    )

    ax = plot_vectorized_observation(observation)

    obstacle_line = next(
        line for line in ax.lines if line.get_label() == "obstacle trajectories"
    )
    assert obstacle_line.get_marker() == "None"


def test_middle_timestamp_sampling_is_seeded_and_avoids_scene_ends() -> None:
    """Automatic plotting samples reproducibly from the central half."""
    timestamps = np.arange(0, 1_000, 100, dtype=np.uint64)

    sampled = sample_middle_timestamp_us(timestamps, seed=17)

    assert sampled == sample_middle_timestamp_us(timestamps, seed=17)
    assert sampled in timestamps[2:8]
