# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import math

import numpy as np
import pytest
from alpasim_runtime.route_generator import (
    RouteGenerator,
    RouteGeneratorMap,
    RouteGeneratorRecorded,
)
from alpasim_utils.artifact import Artifact
from alpasim_utils.geometry import Polyline, Pose
from tests.fixtures import sample_artifact  # noqa: F401

COS_THETA = math.cos(math.radians(30))
SIN_THETA = math.sin(math.radians(30))
STEP = 1.0
IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _make_pose(vec3: np.ndarray) -> Pose:
    """Create a Pose with identity rotation from a 3D position."""
    return Pose(np.asarray(vec3, dtype=np.float32), IDENTITY_QUAT)


@pytest.fixture
def rig_waypoints_in_local():
    # create a set of waypoints that form a straight line at 30 degrees from the x-axis
    num_waypoints = 100
    waypoints = np.zeros((num_waypoints, 3))
    for i in range(num_waypoints):
        waypoints[i] = np.array([COS_THETA * i * STEP, SIN_THETA * i * STEP, 0.0])
    return waypoints


def test_route_generator_invalid_waypoints():
    with pytest.raises(ValueError):
        RouteGeneratorRecorded(np.zeros((1, 3)))


def test_route_generator_nominal_from_origin(rig_waypoints_in_local):
    route_generator = RouteGeneratorRecorded(rig_waypoints_in_local)

    pose_local_to_rig = _make_pose(np.array([0.0, 0.0, 0.0]))
    route_rig = route_generator.generate_route(0, pose_local_to_rig)
    waypoints_rig = route_rig.waypoints

    assert waypoints_rig[0] == pytest.approx(
        [
            0.0,
            0.0,
            0.0,
        ]
    )
    assert waypoints_rig[1] == pytest.approx(
        [
            RouteGenerator.DISTANCE_BETWEEN_WAYPOINTS * COS_THETA,
            RouteGenerator.DISTANCE_BETWEEN_WAYPOINTS * SIN_THETA,
            0.0,
        ]
    )
    EXPECTED_DISTANCE = RouteGenerator.DISTANCE_BETWEEN_WAYPOINTS * (
        RouteGenerator.NUM_WAYPOINTS - 1
    )
    assert waypoints_rig[-1] == pytest.approx(
        [COS_THETA * EXPECTED_DISTANCE, SIN_THETA * EXPECTED_DISTANCE, 0.0]
    )


def test_route_generator_nominal_along_line(rig_waypoints_in_local):
    # sanity check correct waypoints  when pose is along the line
    route_generator = RouteGeneratorRecorded(rig_waypoints_in_local)
    start_position = np.array([COS_THETA * STEP * 10.5, SIN_THETA * STEP * 10.5, 0.0])
    pose_local_to_rig = _make_pose(start_position)
    route_rig = route_generator.generate_route(0, pose_local_to_rig)
    waypoints_rig = route_rig.waypoints

    # Use abs tolerance for float32 precision
    assert waypoints_rig[0] == pytest.approx(
        [
            0.0,
            0.0,
            0.0,
        ],
        abs=1e-5,
    )
    assert waypoints_rig[1] == pytest.approx(
        [
            RouteGenerator.DISTANCE_BETWEEN_WAYPOINTS * COS_THETA,
            RouteGenerator.DISTANCE_BETWEEN_WAYPOINTS * SIN_THETA,
            0.0,
        ],
        abs=1e-5,
    )
    assert np.linalg.norm(waypoints_rig[-1] - waypoints_rig[0]) == pytest.approx(
        RouteGenerator.DISTANCE_BETWEEN_WAYPOINTS * (RouteGenerator.NUM_WAYPOINTS - 1),
        abs=1.0e-3,
    )


def test_route_generator_nominal_off_line(rig_waypoints_in_local):
    # sanity check when we are off the line
    route_generator = RouteGeneratorRecorded(rig_waypoints_in_local)

    start_position = np.array([COS_THETA * STEP * 10, SIN_THETA * STEP * 10, 0.0])
    position_with_offset = start_position + np.array([STEP, 0.0, 0.0])  # add offset
    pose_local_to_rig = _make_pose(position_with_offset)

    reference_waypoint_in_rig = np.array(
        [-STEP * (1.0 - COS_THETA**2), STEP * COS_THETA * SIN_THETA, 0.0]
    )
    expected_second_waypoint = reference_waypoint_in_rig + np.array(
        [
            COS_THETA * RouteGenerator.DISTANCE_BETWEEN_WAYPOINTS,
            SIN_THETA * RouteGenerator.DISTANCE_BETWEEN_WAYPOINTS,
            0.0,
        ]
    )

    route_rig = route_generator.generate_route(0, pose_local_to_rig)
    waypoints_rig = route_rig.waypoints
    assert waypoints_rig[0] == pytest.approx(reference_waypoint_in_rig)
    assert waypoints_rig[1] == pytest.approx(expected_second_waypoint)

    assert np.linalg.norm(waypoints_rig[-1] - waypoints_rig[0]) == pytest.approx(
        RouteGenerator.DISTANCE_BETWEEN_WAYPOINTS * (RouteGenerator.NUM_WAYPOINTS - 1),
        abs=1.0e-3,
    )


def test_route_generator_no_extrapolation(rig_waypoints_in_local):
    route_generator = RouteGeneratorRecorded(rig_waypoints_in_local)
    # sanity check truncation at the end of the route
    start_position = np.array([COS_THETA * STEP * 90, SIN_THETA * STEP * 90, 0.0])
    pose_local_to_rig = _make_pose(start_position)

    route_rig = route_generator.generate_route(0, pose_local_to_rig)
    assert len(route_rig) == 3  # 90m, 94 m into the route, and 98 m into the route

    # check that padding works properly here
    route_rig = RouteGenerator.prepare_for_policy(route_rig)
    assert len(route_rig) == RouteGenerator.NUM_WAYPOINTS
    assert np.all(np.isnan(route_rig.waypoints[3:]))


def test_route_generator_at_end(rig_waypoints_in_local):
    route_generator = RouteGeneratorRecorded(rig_waypoints_in_local)
    # sanity check truncation at the end of the route
    pose_local_to_rig = _make_pose(rig_waypoints_in_local[-1])

    route_rig = route_generator.generate_route(0, pose_local_to_rig)
    assert len(route_rig) == 1


def test_route_generator_passed_end(rig_waypoints_in_local):
    route_generator = RouteGeneratorRecorded(rig_waypoints_in_local)
    # last + delta in same direction as last segment
    point = 2 * rig_waypoints_in_local[-1] - rig_waypoints_in_local[-2]
    pose_local_to_rig = _make_pose(point)

    route_rig = route_generator.generate_route(0, pose_local_to_rig)
    assert len(route_rig) == 0

    # also check that padding works properly here
    route_rig = RouteGenerator.prepare_for_policy(route_rig)
    assert len(route_rig) == RouteGenerator.NUM_WAYPOINTS
    assert np.all(np.isnan(route_rig.waypoints))


def test_route_generator_map(sample_artifact):  # noqa: F811
    route_generator = RouteGeneratorMap(
        sample_artifact.rig.trajectory.positions, sample_artifact.map
    )

    pose_local_to_rig = _make_pose(np.array([0.0, 0.0, 0.0]))
    route_rig = route_generator.generate_route(0, pose_local_to_rig)
    waypoints_rig = route_rig.waypoints
    waypoints_local = waypoints_rig.copy()  # equivalent since we are at the origin

    # per inspection of the map
    expected_first_waypoint = np.array([4.21, -0.49, 0.0])
    assert waypoints_rig[1] == pytest.approx(expected_first_waypoint, abs=1.0e-2)

    # shift in the y direction
    pose_local_to_rig = Pose(
        position=pose_local_to_rig.vec3 + np.array([0.0, 1.0, 0.0], dtype=np.float32),
        quaternion=IDENTITY_QUAT,
    )
    route_rig = route_generator.generate_route(0, pose_local_to_rig)
    waypoints_rig = route_rig.waypoints
    expected_first_waypoint += np.array([0.0, -1.0, 0.0])
    assert waypoints_rig[1] == pytest.approx(expected_first_waypoint, abs=1.0e-2)

    # move along the route and check that waypoints are extended past the end of the scene
    for i in range(len(waypoints_local)):
        pose_local_to_rig = _make_pose(waypoints_local[i])
        route_rig = route_generator.generate_route(0, pose_local_to_rig)
        assert len(route_rig) == RouteGenerator.NUM_WAYPOINTS
        total_distance = route_rig.total_length
        assert total_distance == pytest.approx(
            RouteGenerator.DISTANCE_BETWEEN_WAYPOINTS
            * (RouteGenerator.NUM_WAYPOINTS - 1),
            abs=0.5,
        )


def test_find_consistent_upcoming_lane_nominal():
    upcoming_lane_ids = [["25313575", "66704058", "25313576"]] * 3
    upcoming_lane_ids.append(["66704058"])
    assert (
        RouteGeneratorMap.find_consistent_upcoming_lane_id(upcoming_lane_ids)
        == "66704058"
    )


def test_find_consistent_upcoming_lane_id_all_duplicates():
    """
    Edge case where all upcoming lane ids are the same
    """
    upcoming_lane_ids = [["25313575", "66704058", "25313576"]] * 20
    assert (
        RouteGeneratorMap.find_consistent_upcoming_lane_id(upcoming_lane_ids)
        == upcoming_lane_ids[0][0]
    )


def test_route_generator_map_sanity_off_route():
    usdz_file = "tests/data/route_generator_sanity/sanity_off_route.usdz"
    artifact = Artifact(source=usdz_file)
    # expect exception when the trajectory is far from route
    # in this case, this is because the vehicle drove off the covered map
    with pytest.raises(ValueError) as e:
        RouteGeneratorMap(artifact.rig.trajectory.positions, artifact.map)
    assert "sanity check" in str(e.value)


def test_route_generator_map_sanity_fold_back():
    """
    There were some garbage maps that had next lane segments that folded back on themselves.
    This test ensures that we catch those cases.
    """

    # A portion of data from a bad map that had foldbacks
    bad_waypoints = np.array(
        [
            [2.75341370e01, -4.40550225e00, 0.00000000e00],
            [3.16241059e01, -5.39478687e00, 0.00000000e00],
            [3.56524168e01, -6.61415496e00, 0.00000000e00],
            [3.96273980e01, -8.00180967e00, 0.00000000e00],
            [4.36271219e01, -8.66692462e00, 0.00000000e00],
            [4.68018346e01, -1.09475217e01, 0.00000000e00],
            [5.06923587e01, -1.25520484e01, 0.00000000e00],
            [4.77199377e01, -1.12217031e01, 0.00000000e00],
            [4.38291105e01, -9.61351713e00, 0.00000000e00],
            [3.98998194e01, -8.10171129e00, 0.00000000e00],
        ]
    )

    with pytest.raises(ValueError):
        RouteGenerator.sanity_check_waypoints_for_foldback(bad_waypoints)


def test_route_generator_prepare_for_policy():
    """
    Test proper preparation of the polyline for the policy, including padding with NaNs
    when there are insufficient waypoints and truncation when the points aren't spaced
    properly.
    """
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.2, 0.0, 0.0],
            [8.4, 0.0, 0.0],
            [9.0, 0.0, 0.0],  # first bad one at 3rd index
            [13.2, 0.0, 0.0],
            [17.4, 0.0, 0.0],
        ]
    )
    polyline = Polyline(points=points)
    original_points = points.copy()

    result = RouteGenerator.prepare_for_policy(polyline)
    assert len(result) == RouteGenerator.NUM_WAYPOINTS
    for i in range(3):
        assert result.waypoints[i] == pytest.approx(original_points[i])
    assert np.all(np.isnan(result.waypoints[3:]))
