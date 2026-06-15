# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Tests for eval.data module, particularly RenderableTrajectory.

These tests guard against regressions when the underlying Trajectory class changes,
ensuring that RenderableTrajectory properly inherits from and initializes Trajectory.
"""

from typing import cast

import numpy as np
import pytest
from alpasim_grpc.v0.egodriver_pb2 import DriveResponse
from alpasim_utils.geometry import Pose, Trajectory

from eval import data as eval_data
from eval.data import RAABB, DriverResponseAtTime, DriverResponses, RenderableTrajectory


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


class TestDriverResponsesGetForTimeEmpty:
    """Regression: get_driver_response_for_time must handle empty response lists.

    Sessions that abort during force_gt warmup with ``step_subsample_rate > 1``
    can reach the eval pipeline with ``timestamps_us`` and/or
    ``query_times_us`` empty — no policy-driven driver response was ever
    recorded before the abort.  Several scorers (MinADE, PlanDeviation,
    Safety) iterate over the metric grid and call this method; if it crashes
    on an empty response list, every scorer dies with IndexError and the
    whole eval subprocess is taken out as BrokenProcessPool.
    """

    def _empty_driver_responses(self, sample_raabb: RAABB) -> DriverResponses:
        ego_traj = RenderableTrajectory(
            timestamps_us=np.array([0], dtype=np.uint64),
            positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            quaternions=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            raabb=sample_raabb,
        )
        return DriverResponses(
            ego_coords_rig_to_aabb_center=Pose.identity(),
            ego_trajectory_local=ego_traj,
        )

    def test_get_for_time_empty_returns_none_now(self, sample_raabb: RAABB) -> None:
        dr = self._empty_driver_responses(sample_raabb)
        assert dr.get_driver_response_for_time(100_000, "now") is None

    def test_get_for_time_empty_returns_none_query(self, sample_raabb: RAABB) -> None:
        dr = self._empty_driver_responses(sample_raabb)
        assert dr.get_driver_response_for_time(100_000, "query") is None

    def test_get_for_time_past_end_returns_none(self, sample_raabb: RAABB) -> None:
        """``time`` beyond the last recorded entry returns None (no IndexError).

        Previously this crashed with ``IndexError`` on
        ``timestamps_to_search[idx]`` when ``idx == len(...)`` and the
        corner-case ``query_times_us[-1]`` guard didn't match.
        """
        ego_traj = RenderableTrajectory(
            timestamps_us=np.array([0], dtype=np.uint64),
            positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            quaternions=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            raabb=sample_raabb,
        )
        dr = DriverResponses(
            ego_coords_rig_to_aabb_center=Pose.identity(),
            ego_trajectory_local=ego_traj,
            timestamps_us=[100_000, 200_000],
            query_times_us=[100_000, 200_000],
            per_timestep_driver_responses=[],  # unused on the no-response path
        )
        # Past last entry, but not matching the query_times_us[-1] corner case.
        assert dr.get_driver_response_for_time(500_000, "now") is None


class TestDriverResponsesGetForTimeFallback:
    """Regression: renderers can hold the previous response at off-cadence frames.

    The video animation iterates EGO trajectory timestamps, which run at
    ``pose_reporting_interval_us`` (e.g. 100ms). Driver responses are recorded
    at ``control_timestep_us`` (e.g. 200ms). A frame timestamp landing between
    two responses has no exact match. Exact lookup stays exact for metrics;
    renderers opt into holding the most recent response.
    """

    def _responses(self, sample_raabb: RAABB) -> DriverResponses:
        ego_traj = RenderableTrajectory(
            timestamps_us=np.array([0], dtype=np.uint64),
            positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            quaternions=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            raabb=sample_raabb,
        )
        r0 = cast(DriverResponseAtTime, object())
        r1 = cast(DriverResponseAtTime, object())
        dr = DriverResponses(
            ego_coords_rig_to_aabb_center=Pose.identity(),
            ego_trajectory_local=ego_traj,
            timestamps_us=[0, 200_000],  # responses at 5Hz replan cadence
            query_times_us=[0, 200_000],
            per_timestep_driver_responses=[r0, r1],
        )
        return dr, r0, r1

    def test_exact_lookup_returns_none_off_cadence(self, sample_raabb: RAABB) -> None:
        dr, r0, r1 = self._responses(sample_raabb)
        # Cadence-matched frames hit the exact response.
        assert dr.get_driver_response_for_time(0, "now") is r0
        assert dr.get_driver_response_for_time(200_000, "now") is r1
        # Metric-style exact lookup does not reinterpret an old response as new.
        assert dr.get_driver_response_for_time(100_000, "now") is None

    def test_previous_fallback_returns_at_or_before(self, sample_raabb: RAABB) -> None:
        dr, r0, r1 = self._responses(sample_raabb)
        assert dr.get_driver_response_for_time(-1, "now", fallback="previous") is None
        assert dr.get_driver_response_for_time(0, "now", fallback="previous") is r0
        # 100_000 is a sub-control-step ego frame between responses 0 and 200_000.
        assert (
            dr.get_driver_response_for_time(100_000, "now", fallback="previous") is r0
        )
        assert (
            dr.get_driver_response_for_time(200_000, "now", fallback="previous") is r1
        )
        assert (
            dr.get_driver_response_for_time(500_000, "now", fallback="previous") is r1
        )


def _drive_response_with_debug(debug_payload: bytes) -> DriveResponse:
    response = DriveResponse()
    for i in range(2):
        pose_at_time = response.trajectory.poses.add()
        pose_at_time.timestamp_us = i * 100_000
        pose_at_time.pose.vec.x = float(i)
        pose_at_time.pose.vec.y = 0.0
        pose_at_time.pose.vec.z = 0.0
        pose_at_time.pose.quat.x = 0.0
        pose_at_time.pose.quat.y = 0.0
        pose_at_time.pose.quat.z = 0.0
        pose_at_time.pose.quat.w = 1.0
    response.debug_info.unstructured_debug_info = debug_payload
    return response


class TestDriverResponseDebugInfo:
    def test_extract_debug_extra_does_not_unpickle_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = _drive_response_with_debug(b"not-a-trusted-pickle")

        def fail_if_called(_payload: bytes) -> object:
            raise AssertionError("pickle.loads should not be called")

        monkeypatch.setattr(eval_data.pickle, "loads", fail_if_called)

        assert DriverResponseAtTime._extract_debug_extra(response) is None

    def test_from_drive_response_does_not_unpickle_by_default(
        self, sample_raabb: RAABB, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = _drive_response_with_debug(b"not-a-trusted-pickle")

        def fail_if_called(_payload: bytes) -> object:
            raise AssertionError("pickle.loads should not be called")

        monkeypatch.setattr(eval_data.pickle, "loads", fail_if_called)

        parsed = DriverResponseAtTime.from_drive_response(
            response,
            now_time_us=0,
            query_time_us=100_000,
            ego_raabb=sample_raabb,
            ego_coords_rig_to_aabb_center=Pose.identity(),
        )

        assert parsed.command_name is None
        assert parsed.reasoning_text is None

    def test_driver_responses_do_not_unpickle_by_default(
        self, sample_raabb: RAABB, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = _drive_response_with_debug(b"not-a-trusted-pickle")

        def fail_if_called(_payload: bytes) -> object:
            raise AssertionError("pickle.loads should not be called")

        monkeypatch.setattr(eval_data.pickle, "loads", fail_if_called)
        ego_traj = RenderableTrajectory.create_empty_with_bbox(sample_raabb)
        driver_responses = DriverResponses(
            ego_coords_rig_to_aabb_center=Pose.identity(),
            ego_trajectory_local=ego_traj,
        )

        driver_responses.add_drive_response(
            response,
            now_time_us=0,
            query_time_us=100_000,
        )

        parsed = driver_responses.per_timestep_driver_responses[0]
        assert parsed.command_name is None
        assert parsed.reasoning_text is None

    def test_can_parse_unstructured_debug_info_when_enabled(
        self, sample_raabb: RAABB
    ) -> None:
        response = _drive_response_with_debug(
            eval_data.pickle.dumps({"command_name": "STRAIGHT"})
        )

        parsed = DriverResponseAtTime.from_drive_response(
            response,
            now_time_us=0,
            query_time_us=100_000,
            ego_raabb=sample_raabb,
            ego_coords_rig_to_aabb_center=Pose.identity(),
            parse_unstructured_debug_info=True,
        )

        assert parsed.command_name == "STRAIGHT"

    def test_does_not_unpickle_unstructured_debug_info_when_disabled(
        self, sample_raabb: RAABB, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = _drive_response_with_debug(b"not-a-trusted-pickle")

        def fail_if_called(_payload: bytes) -> object:
            raise AssertionError("pickle.loads should not be called")

        monkeypatch.setattr(eval_data.pickle, "loads", fail_if_called)

        parsed = DriverResponseAtTime.from_drive_response(
            response,
            now_time_us=0,
            query_time_us=100_000,
            ego_raabb=sample_raabb,
            ego_coords_rig_to_aabb_center=Pose.identity(),
            parse_unstructured_debug_info=False,
        )

        assert parsed.command_name is None
        assert parsed.reasoning_text is None
