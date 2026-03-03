# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import numpy as np
import pytest
from alpasim_utils.geometry import (
    Polyline,
    Pose,
    Trajectory,
    polyline_from_grpc,
    polyline_to_grpc_route,
)
from numpy.testing import assert_almost_equal


@pytest.fixture
def simple_polyline() -> Polyline:
    """Create a simple square polyline for testing."""

    return Polyline(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
            ]
        )
    )


def test_compute_arc_lengths_basic():
    points = np.array([[0, 0, 0], [3, 4, 0], [3, 8, 0]], dtype=float)
    arc_lengths = Polyline(points).arc_lengths()

    assert arc_lengths.tolist() == [0.0, 5.0, 9.0]


def test_interpolate_along_path_handles_stationary_points():
    points = np.array([[0, 0, 0], [0, 0, 0], [10, 0, 0]], dtype=float)
    distances = np.array([0.0, 5.0, 10.0])

    interpolated = Polyline(points).positions_at(distances)

    assert np.allclose(interpolated[0], [0, 0, 0])
    assert np.allclose(interpolated[1], [5, 0, 0])
    assert np.allclose(interpolated[2], [10, 0, 0])


def test_project_point_to_polyline_mid_segment():
    points = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0]], dtype=float)
    projected, segment_idx, distance_along = Polyline(points).project_point(
        np.array([4, 3, 0], dtype=float)
    )

    assert segment_idx == 0
    assert distance_along == pytest.approx(4.0)
    assert np.allclose(projected, [4, 0, 0])


def test_remaining_polyline_from_point_at_route_end():
    points = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
    remaining_polyline, (proj_point, seg_idx, dist_along) = Polyline(
        points
    ).remaining_from_point(np.array([20, 0, 0], dtype=float))

    assert remaining_polyline.is_empty
    assert seg_idx == 0


def test_interpolate_along_path_2d():
    points = np.array([[0, 0], [3, 4], [3, 8]], dtype=float)
    distances = np.array([0.0, 5.0, 9.0])

    interpolated = Polyline(points).positions_at(distances)

    assert interpolated.shape == (3, 2)
    assert np.allclose(interpolated[1], [3.0, 4.0])


def test_remaining_polyline_from_point_2d():
    points = np.array([[0, 0], [10, 0]], dtype=float)
    remaining_polyline, (proj_point, seg_idx, dist_along) = Polyline(
        points
    ).remaining_from_point(np.array([5, 5], dtype=float))

    assert remaining_polyline.waypoints.shape == (2, 2)
    assert np.allclose(remaining_polyline.waypoints[0], [5.0, 0.0])
    assert seg_idx == 0


def test_polyline_properties(simple_polyline: Polyline) -> None:
    assert len(simple_polyline) == 4
    assert simple_polyline.total_length == pytest.approx(30.0)
    assert len(simple_polyline.segment_lengths) == 3
    assert_almost_equal(simple_polyline.segment_lengths, [10.0, 10.0, 10.0])


def test_polyline_transform() -> None:
    polyline_obj = Polyline(points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    transform = Pose(
        np.array([5, 5, 0], dtype=np.float32),
        np.array([0, 0, 0, 1], dtype=np.float32),
    )

    transformed = polyline_obj.transform(transform)
    assert_almost_equal(transformed.waypoints[0], [5, 5, 0])
    assert_almost_equal(transformed.waypoints[1], [6, 5, 0])


def test_polyline_trajectory_conversion() -> None:
    timestamps = np.array([0, 1_000_000, 2_000_000], dtype=np.uint64)
    positions = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 0]], dtype=np.float32)
    poses = [
        Pose(positions[i], np.array([0, 0, 0, 1], dtype=np.float32))
        for i in range(len(positions))
    ]
    traj = Trajectory.from_poses(timestamps=timestamps, poses=poses)

    polyline_obj = traj.to_polyline()
    assert len(polyline_obj) == 3
    assert_almost_equal(polyline_obj.waypoints, positions)


def test_polyline_create_empty() -> None:
    empty_3d = Polyline.create_empty()
    assert empty_3d.is_empty
    assert empty_3d.dimension == 3

    empty_2d = Polyline.create_empty(dimension=2)
    assert empty_2d.is_empty
    assert empty_2d.dimension == 2

    with pytest.raises(ValueError):
        Polyline.create_empty(dimension=4)


def test_polyline_grpc_conversion() -> None:
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    polyline_obj = Polyline(points=points)
    timestamp_us = 123_456_789

    grpc_route = polyline_to_grpc_route(polyline_obj, timestamp_us)
    assert len(grpc_route.waypoints) == 2
    assert grpc_route.timestamp_us == timestamp_us
    assert grpc_route.waypoints[0].x == 1.0
    assert grpc_route.waypoints[1].z == 6.0

    parsed = polyline_from_grpc(grpc_route)
    assert_almost_equal(parsed.waypoints, points)


def test_polyline_project_point() -> None:
    polyline_obj = Polyline(points=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))

    projected, segment_idx, distance_along = polyline_obj.project_point(
        np.array([5.0, 2.0, 0.0])
    )
    assert np.allclose(projected, [5, 0, 0])
    assert segment_idx == 0
    assert distance_along == pytest.approx(5.0)

    projected, segment_idx, distance_along = polyline_obj.project_point(
        np.array([15.0, 0.0, 0.0])
    )
    assert np.allclose(projected, [10, 0, 0])
    assert distance_along == pytest.approx(10.0)

    projected, segment_idx, distance_along = polyline_obj.project_point(
        np.array([-5.0, 0.0, 0.0])
    )
    assert np.allclose(projected, [0, 0, 0])
    assert distance_along == pytest.approx(0.0)


def test_polyline_resample_from_point_cases() -> None:
    polyline_obj = Polyline(
        points=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    )

    at_end = polyline_obj.resample_from_point(
        start_point=np.array([20.0, 0.0, 0.0]), spacing=5.0, n_points=5
    )
    assert len(at_end) == 1
    assert np.allclose(at_end.waypoints[0], [20, 0, 0])

    past_end = polyline_obj.resample_from_point(
        start_point=np.array([25.0, 0.0, 0.0]), spacing=5.0, n_points=5
    )
    assert len(past_end) == 0

    extended_polyline = Polyline(
        points=np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]]
        )
    )

    between = extended_polyline.resample_from_point(
        start_point=np.array([5.0, 0.0, 0.0]), spacing=10.0, n_points=3
    )
    assert len(between) == 3
    assert np.allclose(between.waypoints[0], [5, 0, 0])
    assert np.allclose(between.waypoints[1], [15, 0, 0])
    assert np.allclose(between.waypoints[2], [25, 0, 0])

    off_path = extended_polyline.resample_from_point(
        start_point=np.array([5.0, 5.0, 0.0]), spacing=5.0, n_points=4
    )
    assert len(off_path) == 4
    assert np.allclose(off_path.waypoints[0], [5, 0, 0])
    assert np.allclose(off_path.waypoints[-1], [20, 0, 0])


def test_polyline_get_cumulative_distances_from_point() -> None:
    polyline_obj = Polyline(
        points=np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]]
        )
    )

    cumulative, distance_to_projection = (
        polyline_obj.get_cumulative_distances_from_point(np.array([5.0, 0.0, 0.0]))
    )

    assert distance_to_projection == pytest.approx(5.0)
    assert len(cumulative) == 4
    assert cumulative[0] == 0.0
    assert cumulative[1] == pytest.approx(5.0)
    assert cumulative[2] == pytest.approx(15.0)
    assert cumulative[3] == pytest.approx(25.0)


def test_polyline_append() -> None:
    polyline1 = Polyline(points=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
    polyline2 = Polyline(points=np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]))

    combined = polyline1.append(polyline2)
    assert len(combined) == 4
    assert np.allclose(combined.waypoints[0], [0, 0, 0])
    assert np.allclose(combined.waypoints[-1], [20, 0, 0])


def test_polyline_append_dimension_mismatch() -> None:
    polyline1 = Polyline(points=np.array([[0.0, 0.0]]))
    polyline2 = Polyline(points=np.array([[0.0, 0.0, 0.0]]))

    with pytest.raises(ValueError):
        polyline1.append(polyline2)


def test_polyline_append_empty() -> None:
    polyline_obj = Polyline(points=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
    empty = Polyline.create_empty()

    result = polyline_obj.append(empty)
    assert len(result) == 2
    assert np.allclose(result.waypoints, polyline_obj.waypoints)

    result = empty.append(polyline_obj)
    assert len(result) == 2
    assert np.allclose(result.waypoints, polyline_obj.waypoints)


def test_polyline_clip() -> None:
    polyline_obj = Polyline(
        points=np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]]
        )
    )

    clipped = polyline_obj.clip(1, 3)
    assert len(clipped) == 2
    assert np.allclose(clipped.waypoints[0], [10, 0, 0])
    assert np.allclose(clipped.waypoints[1], [20, 0, 0])

    clipped = polyline_obj.clip(end=2)
    assert len(clipped) == 2
    assert np.allclose(clipped.waypoints[0], [0, 0, 0])
    assert np.allclose(clipped.waypoints[1], [10, 0, 0])

    clipped = polyline_obj.clip(start=2)
    assert len(clipped) == 2
    assert np.allclose(clipped.waypoints[0], [20, 0, 0])
    assert np.allclose(clipped.waypoints[1], [30, 0, 0])


def test_polyline_zero_out_z() -> None:
    polyline_obj = Polyline(
        points=np.array([[0.0, 0.0, 1.0], [10.0, 0.0, 2.0], [20.0, 0.0, 3.0]])
    )

    zeroed = polyline_obj.zero_out_z()
    assert np.allclose(zeroed.waypoints[:, :2], polyline_obj.waypoints[:, :2])
    assert np.allclose(zeroed.waypoints[:, 2], 0.0)
    assert np.allclose(polyline_obj.waypoints[:, 2], [1, 2, 3])


def test_polyline_single_waypoint() -> None:
    polyline_obj = Polyline(points=np.array([[5.0, 5.0, 0.0]]))

    assert len(polyline_obj) == 1
    assert polyline_obj.total_length == 0.0
    assert len(polyline_obj.segment_lengths) == 0

    projected, segment_idx, distance_along = polyline_obj.project_point(
        np.array([0.0, 0.0, 0.0])
    )
    assert np.allclose(projected, [5, 5, 0])
    assert segment_idx == 0
    assert distance_along == 0.0

    resampled = polyline_obj.resample_from_point(
        np.array([0.0, 0.0, 0.0]), spacing=1.0, n_points=5
    )
    assert len(resampled) == 1
    assert np.allclose(resampled.waypoints[0], [5, 5, 0])


def test_polyline_degenerate_segments() -> None:
    polyline_obj = Polyline(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
            ]
        )
    )

    projected, _, _ = polyline_obj.project_point(np.array([10.0, 5.0, 0.0]))
    assert np.allclose(projected, [10, 0, 0])
    assert polyline_obj.total_length == pytest.approx(20.0)


def test_polyline_rejects_1d_array() -> None:
    """Test that 1D arrays are rejected with informative error."""
    with pytest.raises(TypeError, match="1D"):
        Polyline(points=np.array([1.0, 2.0, 3.0]))


def test_polyline_rejects_int_array() -> None:
    """Test that integer arrays are rejected with informative error."""
    with pytest.raises(TypeError, match="int"):
        Polyline(points=np.array([[0, 0, 0], [1, 0, 0]]))


def test_polyline_remaining_from_empty() -> None:
    empty = Polyline.create_empty()
    remaining, (proj_point, seg_idx, dist_along) = empty.remaining_from_point(
        np.zeros(3)
    )

    assert remaining.is_empty
    assert np.allclose(proj_point, np.zeros(3))
    assert seg_idx == 0
    assert dist_along == 0.0


def test_polyline_transform_with_rotation() -> None:
    polyline_obj = Polyline(points=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    quat_z_90 = np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)], dtype=np.float32)
    transform = Pose(np.array([0, 0, 0], dtype=np.float32), quat_z_90)

    transformed = polyline_obj.transform(transform)
    assert_almost_equal(transformed.waypoints[0], [0, 1, 0], decimal=5)
    assert_almost_equal(transformed.waypoints[1], [0, 2, 0], decimal=5)


def test_polyline_get_cumulative_distances_from_point_off_path() -> None:
    polyline_obj = Polyline(
        points=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    )

    cumulative, distance_to_projection = (
        polyline_obj.get_cumulative_distances_from_point(np.array([5.0, 5.0, 0.0]))
    )

    assert distance_to_projection == pytest.approx(5.0)
    assert len(cumulative) == 3
    assert cumulative[0] == 0.0
    assert cumulative[1] == pytest.approx(5.0)
    assert cumulative[2] == pytest.approx(15.0)


def test_polyline_get_cumulative_distances_from_point_accepts_float32_point() -> None:
    """Verify get_cumulative_distances_from_point accepts float32 inputs (e.g. Pose.vec3).

    Both float32 and float64 arrays are supported.
    """
    polyline_obj = Polyline(
        points=np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
            dtype=np.float32,
        )
    )
    point_float32 = np.array([5.0, 0.0, 0.0], dtype=np.float32)

    cumulative, distance_to_projection = (
        polyline_obj.get_cumulative_distances_from_point(point_float32)
    )

    assert distance_to_projection == pytest.approx(5.0)
    assert len(cumulative) == 4
    assert cumulative[0] == 0.0
    assert cumulative[1] == pytest.approx(5.0)
    assert cumulative[2] == pytest.approx(15.0)
    assert cumulative[3] == pytest.approx(25.0)


def test_polyline_get_cumulative_distances_from_point_rejects_int_dtype_with_clear_error() -> (
    None
):
    """get_cumulative_distances_from_point should reject invalid dtypes with a clear message.

    The point argument should go through type validation (e.g. array_utils)
    so that int or other invalid dtypes produce a helpful error instead of
    a generic PyArray conversion error.
    """
    polyline_obj = Polyline(
        points=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    )
    point_int = np.array([5, 0, 0], dtype=np.int64)

    with pytest.raises(TypeError) as exc_info:
        polyline_obj.get_cumulative_distances_from_point(point_int)

    msg = str(exc_info.value).lower()
    assert "point" in msg
    assert "dtype" in msg or "float" in msg


def test_polyline_project_point_near_segment_midpoint() -> None:
    polyline_obj = Polyline(
        points=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    )
    query = np.array([15.0, 1.0, 0.0])

    projected, segment_idx, _ = polyline_obj.project_point(query)
    assert segment_idx == 1
    assert np.allclose(projected, [15, 0, 0])
    assert np.linalg.norm(projected - query) == pytest.approx(1.0)


def test_polyline_projection_perpendicular() -> None:
    polyline_obj = Polyline(points=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))

    projected, segment_idx, distance_along = polyline_obj.project_point(
        np.array([5.0, 3.0, 0.0])
    )
    assert np.allclose(projected, [5, 0, 0])
    assert segment_idx == 0
    assert distance_along == pytest.approx(5.0)


def test_polyline_2d_support_and_guards() -> None:
    polyline_2d = Polyline(points=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 1.0]]))
    assert polyline_2d.dimension == 2
    assert polyline_2d.total_length == pytest.approx(np.linalg.norm([1, 1]) + 1.0)

    distances = np.linspace(0.0, polyline_2d.total_length, 5)
    sampled = polyline_2d.positions_at(distances)
    assert sampled.shape == (5, 2)

    with pytest.raises(ValueError):
        polyline_2d.transform(
            Pose(
                np.zeros(3, dtype=np.float32), np.array([0, 0, 0, 1], dtype=np.float32)
            )
        )
    with pytest.raises(ValueError):
        polyline_to_grpc_route(polyline_2d, timestamp_us=0)
    with pytest.raises(ValueError):
        polyline_2d.zero_out_z()


def test_polyline_requires_matching_point_dimension_on_projection() -> None:
    polyline_obj = Polyline(points=np.array([[0.0, 0.0], [1.0, 0.0]]))

    with pytest.raises(ValueError):
        polyline_obj.project_point(np.array([0.0, 0.0, 0.0]))


def test_polyline_downsample_with_min_distance_empty() -> None:
    """Test downsampling an empty polyline."""
    polyline = Polyline.create_empty()
    polyline.downsample_with_min_distance(min_distance=5.0)
    assert len(polyline) == 0


def test_polyline_downsample_with_min_distance_basic() -> None:
    """Test basic downsampling with minimum distance."""
    points = np.array(
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [15.0, 0.0, 0.0]]
    )
    polyline = Polyline(points=points.copy())

    # All big enough, should keep all points
    polyline.downsample_with_min_distance(min_distance=3.0)
    assert len(polyline) == 4
    for i in range(4):
        assert np.allclose(polyline.waypoints[i], points[i])

    # Diffs too small, drop a few
    polyline.downsample_with_min_distance(min_distance=6.0)

    assert len(polyline) == 2
    assert np.allclose(polyline.waypoints[0], [0.0, 0.0, 0.0])
    assert np.allclose(polyline.waypoints[1], [10.0, 0.0, 0.0])


def test_polyline_downsample_with_min_distance_irregular_spacing() -> None:
    """Test downsampling with irregularly spaced points."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # Duplicate point
            [10.0, 0.0, 0.0],
            [10.4, 0.0, 0.0],  # Not far enough
            [20.0, 0.0, 0.0],
        ]
    )
    polyline = Polyline(points=points.copy())
    polyline.downsample_with_min_distance(min_distance=0.5)

    assert (
        len(polyline) == 4
    ), f"Expected 4 points after downsampling, got {polyline.waypoints}"
    assert np.allclose(polyline.waypoints[0], points[0])
    assert np.allclose(polyline.waypoints[1], points[1])
    assert np.allclose(polyline.waypoints[2], points[3])
    assert np.allclose(polyline.waypoints[3], points[5])


def test_polyline_points_and_waypoints_return_correct_arrays() -> None:
    """Verify points/waypoints getters return correct (N, D) arrays.

    Regression test: these getters now propagate errors via PyResult instead
    of silently falling back to a zeros array.  Confirm the happy path still
    works for 2D, 3D, and empty polylines, and that `points` and `waypoints`
    are identical.
    """
    # 3D case
    pts_3d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    poly_3d = Polyline(points=pts_3d)
    assert_almost_equal(poly_3d.points, pts_3d)
    assert_almost_equal(poly_3d.waypoints, pts_3d)
    assert poly_3d.points.shape == (2, 3)

    # 2D case
    pts_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    poly_2d = Polyline(points=pts_2d)
    assert_almost_equal(poly_2d.points, pts_2d)
    assert_almost_equal(poly_2d.waypoints, pts_2d)
    assert poly_2d.points.shape == (3, 2)

    # Empty 3D
    empty_3d = Polyline.create_empty(dimension=3)
    assert empty_3d.points.shape == (0, 3)
    assert empty_3d.waypoints.shape == (0, 3)

    # Empty 2D
    empty_2d = Polyline.create_empty(dimension=2)
    assert empty_2d.points.shape == (0, 2)
    assert empty_2d.waypoints.shape == (0, 2)


def test_polyline_from_non_contiguous_array() -> None:
    """Test that Polyline works with non-contiguous array slices.

    This reproduces an issue where slicing a 4-column array to get first 3 columns
    (e.g., x,y,z from x,y,z,heading) creates a non-contiguous view that failed
    when passed to the Rust backend.
    """
    # Create a 4-column array (simulating x, y, z, heading)
    full_array = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.1],
            [2.0, 0.0, 0.0, 1.2],
            [3.0, 0.0, 0.0, 1.3],
            [4.0, 0.0, 0.0, 1.4],
        ]
    )

    # Slice to get rows 2-4 and first 3 columns (non-contiguous view)
    non_contiguous = full_array[2:, 0:3]
    assert not non_contiguous.flags[
        "C_CONTIGUOUS"
    ], "Test setup: array should be non-contiguous"

    # This should work without raising TypeError
    polyline = Polyline(points=non_contiguous)

    assert len(polyline) == 3
    assert polyline.dimension == 3
    assert np.allclose(polyline.waypoints[0], [2.0, 0.0, 0.0])
    assert np.allclose(polyline.waypoints[1], [3.0, 0.0, 0.0])
    assert np.allclose(polyline.waypoints[2], [4.0, 0.0, 0.0])

    # Verify operations work
    projected, segment_idx, _ = polyline.project_point(np.array([2.5, 1.0, 0.0]))
    assert segment_idx == 0
    assert np.allclose(projected, [2.5, 0.0, 0.0])
