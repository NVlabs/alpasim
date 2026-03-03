# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Tests for eval.data module, particularly RenderableTrajectory.

These tests guard against regressions when the underlying Trajectory class changes,
ensuring that RenderableTrajectory properly inherits from and initializes Trajectory.
"""

import numpy as np
import pytest
from alpasim_utils.geometry import Pose, Trajectory

from eval.data import RAABB, RenderableTrajectory


@pytest.fixture
def sample_raabb() -> RAABB:
    """Sample RAABB for vehicle bounding box."""
    return RAABB(size_x=4.5, size_y=2.0, size_z=1.5, corner_radius_m=0.1)


@pytest.fixture
def sample_trajectory() -> Trajectory:
    """Sample trajectory with a few poses."""
    timestamps_us = np.array([0, 100000, 200000], dtype=np.uint64)
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32
    )
    quaternions = np.array([[0.0, 0.0, 0.0, 1.0]] * 3, dtype=np.float32)
    return Trajectory(timestamps_us, positions, quaternions)


class TestRenderableTrajectoryConstruction:
    """Test RenderableTrajectory construction methods.

    These tests are critical for catching regressions when Trajectory's
    implementation changes (e.g., from dataclass to regular class).
    """

    def test_create_empty_with_bbox(self, sample_raabb: RAABB) -> None:
        """Test creating an empty RenderableTrajectory with a bounding box.

        This was the failing case in the original bug where RenderableTrajectory
        as a dataclass couldn't properly inherit from a non-dataclass Trajectory.
        """
        traj = RenderableTrajectory.create_empty_with_bbox(sample_raabb)

        assert traj.is_empty()
        assert len(traj) == 0
        assert traj.raabb == sample_raabb
        assert traj.polygon_artists is None
        assert traj.renderable_linestring is None
        assert traj.fill_color == "black"
        assert traj.fill_alpha == 0.1

    def test_from_trajectory(
        self, sample_trajectory: Trajectory, sample_raabb: RAABB
    ) -> None:
        """Test creating RenderableTrajectory from an existing Trajectory."""
        renderable = RenderableTrajectory.from_trajectory(
            sample_trajectory, sample_raabb
        )

        assert not renderable.is_empty()
        assert len(renderable) == 3
        assert renderable.raabb == sample_raabb
        np.testing.assert_array_equal(
            renderable.timestamps_us, sample_trajectory.timestamps_us
        )
        np.testing.assert_array_almost_equal(
            renderable.positions, sample_trajectory.positions
        )
        np.testing.assert_array_almost_equal(
            renderable.quaternions,
            sample_trajectory.quaternions,
        )

    def test_direct_construction(self, sample_raabb: RAABB) -> None:
        """Test direct construction of RenderableTrajectory."""
        timestamps_us = np.array([0, 100000], dtype=np.uint64)
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
        quaternions = np.array([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=np.float32)

        renderable = RenderableTrajectory(
            timestamps_us=timestamps_us,
            positions=positions,
            quaternions=quaternions,
            raabb=sample_raabb,
            fill_color="red",
            fill_alpha=0.5,
        )

        assert len(renderable) == 2
        assert renderable.raabb == sample_raabb
        assert renderable.fill_color == "red"
        assert renderable.fill_alpha == 0.5

    def test_from_empty_trajectory(self, sample_raabb: RAABB) -> None:
        """Test creating RenderableTrajectory from an empty Trajectory."""
        empty_traj = Trajectory.create_empty()
        renderable = RenderableTrajectory.from_trajectory(empty_traj, sample_raabb)

        assert renderable.is_empty()
        assert renderable.raabb == sample_raabb


class TestRenderableTrajectoryInheritance:
    """Test that RenderableTrajectory properly inherits Trajectory behavior."""

    def test_transform(
        self, sample_trajectory: Trajectory, sample_raabb: RAABB
    ) -> None:
        """Test that transform returns a RenderableTrajectory with preserved RAABB."""
        renderable = RenderableTrajectory.from_trajectory(
            sample_trajectory, sample_raabb
        )

        # Apply a translation transform
        transform = Pose(
            np.array([10.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
        transformed = renderable.transform(transform)

        assert isinstance(transformed, RenderableTrajectory)
        assert transformed.raabb == sample_raabb
        # Check positions are transformed
        np.testing.assert_array_almost_equal(
            transformed.positions[:, 0],
            sample_trajectory.positions[:, 0] + 10.0,
        )

    def test_interpolate_to_timestamps(
        self, sample_trajectory: Trajectory, sample_raabb: RAABB
    ) -> None:
        """Test that interpolation returns a RenderableTrajectory."""
        renderable = RenderableTrajectory.from_trajectory(
            sample_trajectory, sample_raabb
        )

        # Interpolate to a timestamp between existing ones
        target_ts = np.array([50000, 150000], dtype=np.uint64)
        interpolated = renderable.interpolate_to_timestamps(target_ts)

        assert isinstance(interpolated, RenderableTrajectory)
        assert interpolated.raabb == sample_raabb
        assert len(interpolated) == 2

    def test_time_range_us(
        self, sample_trajectory: Trajectory, sample_raabb: RAABB
    ) -> None:
        """Test that time_range_us property works correctly."""
        renderable = RenderableTrajectory.from_trajectory(
            sample_trajectory, sample_raabb
        )

        time_range = renderable.time_range_us
        assert time_range.start == 0
        assert time_range.stop == 200001  # end is exclusive

    def test_is_empty(self, sample_raabb: RAABB) -> None:
        """Test is_empty method for both empty and non-empty trajectories."""
        empty = RenderableTrajectory.create_empty_with_bbox(sample_raabb)
        assert empty.is_empty()

        non_empty = RenderableTrajectory(
            timestamps_us=np.array([0], dtype=np.uint64),
            positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            quaternions=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            raabb=sample_raabb,
        )
        assert not non_empty.is_empty()


class TestRAABB:
    """Test RAABB dataclass."""

    def test_raabb_creation(self) -> None:
        """Test basic RAABB creation."""
        raabb = RAABB(size_x=5.0, size_y=2.5, size_z=1.8, corner_radius_m=0.2)

        assert raabb.size_x == 5.0
        assert raabb.size_y == 2.5
        assert raabb.size_z == 1.8
        assert raabb.corner_radius_m == 0.2
