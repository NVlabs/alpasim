# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
Provides geometry tools for working with vehicle trajectories, expressed as quaternion + translation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Literal, Sequence

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

try:
    import csaps
except ImportError:
    csaps = None
    print(
        "csaps not found, cubic spline approximation will not be available. "
        "Please install csaps to use this functionality."
    )
import numpy as np
from alpasim_grpc.v0 import common_pb2 as grpc_types
from alpasim_utils.polyline import Polyline
from alpasim_utils.qvec import QVec, assert_is_vec3_shape
from scipy import spatial
from scipy.spatial.transform import Rotation as R


@dataclass
class DynamicState:
    angular_velocity: np.ndarray
    linear_velocity: np.ndarray

    def __post_init__(self):
        assert_is_vec3_shape(self.angular_velocity)
        assert_is_vec3_shape(self.linear_velocity)
        assert self.angular_velocity.shape[:-1] == self.linear_velocity.shape[:-1]

    @classmethod
    def create_empty(cls) -> Self:
        return cls(
            angular_velocity=np.zeros((0, 3), dtype=np.float32),
            linear_velocity=np.zeros((0, 3), dtype=np.float32),
        )

    @property
    def batch_size(self) -> tuple[int, ...]:
        return self.linear_velocity.shape[:-1]

    @staticmethod
    def stack(states: Sequence[DynamicState], axis: int = 0) -> DynamicState:
        return DynamicState(
            angular_velocity=np.stack(
                [state.angular_velocity for state in states], axis=axis
            ),
            linear_velocity=np.stack(
                [state.linear_velocity for state in states], axis=axis
            ),
        )

    def _apply(
        self,
        fn: Callable[
            [
                np.ndarray,
            ],
            np.ndarray,
        ],
    ) -> DynamicState:
        return DynamicState(
            angular_velocity=fn(self.angular_velocity),
            linear_velocity=fn(self.linear_velocity),
        )

    def __getitem__(self, index) -> DynamicState:
        return self._apply(lambda arr: arr[index])

    def __len__(self) -> int:
        return self.batch_size[0]

    def __iter__(self) -> Iterator[DynamicState]:
        for i in range(len(self)):
            yield self[i]

    def append(self, other: DynamicState) -> DynamicState:
        if len(self.batch_size) != 1:
            raise ValueError("Can only append to a single state")
        if other.batch_size != ():
            raise ValueError("Can only append a single state")

        other = other[None, :]  # add batch dimension

        return DynamicState(
            angular_velocity=np.concatenate(
                [self.angular_velocity, other.angular_velocity], axis=0
            ),
            linear_velocity=np.concatenate(
                [self.linear_velocity, other.linear_velocity], axis=0
            ),
        )

    @staticmethod
    def from_grpc_state(grpc_state: grpc_types.DynamicState) -> DynamicState:
        return DynamicState(
            angular_velocity=np.array(
                [getattr(grpc_state.angular_velocity, dim) for dim in "xyz"]
            ),
            linear_velocity=np.array(
                [getattr(grpc_state.linear_velocity, dim) for dim in "xyz"]
            ),
        )

    def as_grpc_state(self) -> grpc_types.DynamicState:
        if self.batch_size != ():
            raise ValueError("Can only convert a single state to a grpc state")
        return grpc_types.DynamicState(
            angular_velocity=grpc_types.Vec3(
                x=self.angular_velocity[0],
                y=self.angular_velocity[1],
                z=self.angular_velocity[2],
            ),
            linear_velocity=grpc_types.Vec3(
                x=self.linear_velocity[0],
                y=self.linear_velocity[1],
                z=self.linear_velocity[2],
            ),
        )

    def to_grpc_pose_at_time(
        self, timestamp_us: int, qvec: QVec
    ) -> grpc_types.PoseAtTime:
        return qvec.to_grpc_pose_at_time(timestamp_us, self)

    def as_grpc_states(self) -> Sequence[grpc_types.DynamicState]:
        assert len(self.batch_size) == 1
        return [state.as_grpc_state() for state in self]


@dataclass
class Trajectory:
    timestamps_us: np.ndarray
    poses: QVec

    def __post_init__(self):
        assert self.timestamps_us.ndim == 1
        assert self.timestamps_us.dtype == np.uint64
        assert self.poses.batch_size == self.timestamps_us.shape

        # check strict monotonicity
        delta = self.timestamps_us[1:] - self.timestamps_us[:-1]
        assert (delta > 0).all()

    def __len__(self) -> int:
        return self.timestamps_us.shape[0]

    def is_empty(self) -> bool:
        return len(self) == 0

    @classmethod
    def create_empty(cls) -> Self:
        return cls(
            timestamps_us=np.array([], dtype=np.uint64),
            poses=QVec.create_empty(),
        )

    def __repr__(self) -> str:
        return f"Trajectory(n_poses={self.timestamps_us.shape[0]}, time_range_us={self.time_range_us})"

    def transform(self, transform: QVec, is_relative: bool = False) -> Trajectory:
        """Transforms the trajectory based on `transform`.

        Args:
            transform: The transform to apply to the trajectory.
            is_relative: Whether the transform is a body-fixed transform relative
                to the pose of the vehicle itself (active transformation), meaning
                we need to perform a right multiplication of the poses such that we,
                e.g., rotate the transform's translation into the local frame before
                including it.
        """
        if self.is_empty():
            return self
        if is_relative:
            return Trajectory(self.timestamps_us, self.poses @ transform)
        else:
            return Trajectory(self.timestamps_us, transform[None, :] @ self.poses)

    def clip(self, start_us: int, end_us: int) -> Trajectory:
        """
        Subselect the portion of `self` which is between `start_us` and `end_us` (exclusive).
        Returns an empty trajectory if that selection is out of bounds of `self`.
        """
        assert start_us <= end_us
        if (
            start_us == end_us
            or self.time_range_us.start > end_us
            or self.time_range_us.stop < start_us
        ):
            return Trajectory.create_empty()

        # clamp the input time range to `self.time_range_us`
        start_us = max(start_us, self.time_range_us.start)
        last_timestamp_us = min(end_us, self.time_range_us.stop) - 1

        # Check if start and end are the same (single timestamp case)
        if start_us == last_timestamp_us:
            # Single timestamp case: just interpolate to that one timestamp
            single_pose = self.interpolate_pose(start_us)
            return Trajectory(
                timestamps_us=np.array([start_us], dtype=np.uint64),
                poses=QVec(
                    vec3=single_pose.vec3[np.newaxis, :],
                    quat=single_pose.quat[np.newaxis, :],
                ),
            )

        # interpolate the start and end poses, retain the poses in between
        first_pose, last_pose = self.interpolate_to_timestamps(
            np.array([start_us, last_timestamp_us], dtype=np.uint64)
        ).poses
        is_between_start_and_end = (self.timestamps_us > start_us) & (
            self.timestamps_us < last_timestamp_us
        )
        poses = [first_pose, *list(self.poses[is_between_start_and_end]), last_pose]
        timestamps_us = [
            start_us,
            *self.timestamps_us[is_between_start_and_end],
            last_timestamp_us,
        ]

        return Trajectory(
            timestamps_us=np.array(timestamps_us, dtype=np.uint64),
            poses=QVec.stack(poses),
        )

    @property
    def time_range_us(self) -> range:
        """The range of timestamps in `self`.

        Note, for testing `time in self.time_range_us`, make sure that `time` is
        a python integer and not a numpy scalar/array, which is _very_ slow.
        """
        if self.is_empty():
            return range(0, 0)

        return range(int(self.timestamps_us[0]), int(self.timestamps_us[-1]) + 1)

    @property
    def last_pose(self) -> QVec:
        return self.poses[-1]

    def update_absolute(self, timestamp: int, pose: QVec) -> None:
        if not self.is_empty():
            assert timestamp > self.time_range_us.stop
        self.timestamps_us = np.concatenate(
            [self.timestamps_us, np.array([timestamp], dtype=np.uint64)], axis=0
        )
        self.poses = self.poses.append(pose)

    def update_relative(self, timestamp: int, pose_delta: QVec) -> None:
        assert timestamp > self.time_range_us.stop
        self.update_absolute(timestamp, self.poses[-1] @ pose_delta)

    def interpolate_to_timestamps(self, ts_target: np.ndarray) -> Trajectory:
        if ts_target.dtype != np.uint64:
            raise TypeError(f"Expected np.uint64 got {ts_target.dtype=}.")

        if self.is_empty():
            raise ValueError("Trying to interpolate on an empty trajectory.")

        is_in_range = (ts_target >= self.time_range_us.start) & (
            ts_target < self.time_range_us.stop
        )
        if not is_in_range.all():
            raise ValueError(
                f"Interpolate @ {ts_target[~is_in_range]} outside of {self.time_range_us}."
            )

        if self.timestamps_us.shape == (1,):
            # Slerp will fail with a single pose. Since we already checked that all queries are
            # in range, we can just return that single pose replicated for each query
            poses = QVec.stack([self.last_pose] * ts_target.shape[0])
        else:
            slerp = spatial.transform.Slerp(
                self.timestamps_us, R.from_quat(self.poses.quat)
            )(ts_target).as_quat()

            lerp = np.stack(
                [
                    np.interp(ts_target, self.timestamps_us, vec_dim)
                    for vec_dim in self.poses.vec3.T
                ],
                axis=1,
            )

            poses = QVec(
                vec3=lerp,
                quat=slerp,
            )

        return Trajectory(timestamps_us=ts_target, poses=poses)

    def interpolate_pose(self, at_us: int) -> QVec:
        return self.interpolate_to_timestamps(np.array([at_us], dtype=np.uint64)).poses[
            0
        ]

    def interpolate_delta(self, start_us: int, end_us: int) -> QVec:
        interp = self.interpolate_to_timestamps(
            np.array([start_us, end_us], dtype=np.uint64)
        )
        return interp.poses[0].inverse() @ interp.poses[1]

    @staticmethod
    def from_grpc(trajectory: grpc_types.Trajectory) -> Trajectory:
        if len(trajectory.poses) == 0:
            return Trajectory.create_empty()

        timestamps_us = np.array(
            [p.timestamp_us for p in trajectory.poses], dtype=np.uint64
        )
        poses = QVec.stack(
            [QVec.from_grpc_pose(p.pose) for p in trajectory.poses], axis=0
        )
        return Trajectory(timestamps_us=timestamps_us, poses=poses)

    def to_grpc(self) -> grpc_types.Trajectory:
        poses = self.poses.as_grpc_poses()
        return grpc_types.Trajectory(
            poses=[
                grpc_types.PoseAtTime(
                    timestamp_us=ts,
                    pose=pose,
                )
                for ts, pose in zip(
                    self.timestamps_us,
                    poses,
                )
            ]
        )

    def clone(self) -> Trajectory:
        return Trajectory(self.timestamps_us.copy(), self.poses.clone())

    def _centered_diff_derivatives(
        self, arr: np.ndarray, timestamps_us: np.ndarray
    ) -> np.ndarray:
        """Computes derivatives using finite differences.

        Returns time derivatives using centered differences for middle points and
        forward/backward differences for endpoints.

        Args:
            arr: [N, ...] array of values to differentiate
            timestamps_us: [N] array of time steps in microseconds
        """
        if arr.shape[0] < 2:
            raise ValueError("Not enough poses to compute velocities")

        timestamps = timestamps_us / 1e6

        for _ in range(arr.ndim - 1):
            timestamps = timestamps[..., None]

        derivatives = np.zeros_like(arr)

        # Forward difference for first point
        derivatives[0] = (arr[1] - arr[0]) / (timestamps[1] - timestamps[0])

        # Centered differences for middle points
        if len(self) > 2:
            deltas = arr[2:] - arr[:-2]
            delta_times = timestamps[2:] - timestamps[:-2]
            derivatives[1:-1] = deltas / delta_times

        # Backward difference for last point
        derivatives[-1] = (arr[-1] - arr[-2]) / (timestamps[-1] - timestamps[-2])

        return derivatives

    def _forward_diff_derivatives(
        self, arr: np.ndarray, timestamps_us: np.ndarray
    ) -> np.ndarray:
        """Computes derivatives using forward differences.

        Returns time derivatives using forward differences, except for the last
        point which is computed using backward differences (effectively a
        backward difference at the last point).
        """
        if arr.shape[0] < 2:
            raise ValueError("Not enough poses to compute velocities")

        timestamps = timestamps_us / 1e6

        for _ in range(arr.ndim - 1):
            timestamps = timestamps[..., None]

        derivatives = np.zeros_like(arr)
        derivatives[:-1] = (arr[1:] - arr[:-1]) / (timestamps[1:] - timestamps[:-1])
        derivatives[-1] = (arr[-1] - arr[-2]) / (timestamps[-1] - timestamps[-2])

        return derivatives

    def _cubic_spline_approximation(
        self,
        arr: np.ndarray,
        timestamps_us: np.ndarray,
        deriv: int = 1,
        smoothing_factor: float | None = None,
    ) -> np.ndarray:
        """Computes derivatives using a cubic spline approximation.

        Args:
            arr: [N, D] array of values to differentiate
            timestamps_us: [N] array of time steps in microseconds
            deriv: Order of derivative to compute
            smoothing_factor: Smoothing factor for the cubic spline. Leave as
                `None` to use the default value.
        """

        assert arr.ndim <= 2
        assert arr.shape[0] == timestamps_us.shape[0]
        if csaps is None:
            raise ImportError(
                "csaps is not installed. Please install csaps to use cubic spline approximation."
            )

        if arr.ndim == 1:
            arr = arr[..., None]

        # The default smoothing values works based on distances between data
        # points. Hence we pass in the time in seconds, not us, to get a
        # reasonable smoothing factor.
        css = csaps.CubicSmoothingSpline(
            timestamps_us / 1e6,
            np.moveaxis(arr, 0, 1),  # Expects time in last dimension
            normalizedsmooth=True,
            smooth=smoothing_factor,
        )

        return np.moveaxis(css(timestamps_us / 1e6, nu=deriv), 0, 1).squeeze()

    def _derivative(
        self,
        arr: np.ndarray,
        timestamps_us: np.ndarray,
        deriv: int,
        method: Literal["cubic", "forward", "centered"] = "cubic",
        **kwargs: Any,
    ) -> np.ndarray:
        """Helper method for computing derivatives using different methods.

        Args:
            arr: [N, D] array of values to differentiate
            timestamps_us: [N] array of time steps in microseconds
            method: The method to use for computing the derivatives.
            deriv: The order of the derivative to compute.
            **kwargs: Additional arguments to pass to the cubic spline
                approximation.
        """
        if method == "cubic":
            arr = self._cubic_spline_approximation(
                arr, timestamps_us, deriv=deriv, **kwargs
            )
        elif method == "forward":
            for _ in range(deriv):
                arr = self._forward_diff_derivatives(arr, timestamps_us)
        elif method == "centered":
            for _ in range(deriv):
                arr = self._centered_diff_derivatives(arr, timestamps_us)
        else:
            raise ValueError(f"Invalid method: {method}")
        return arr

    def velocities(
        self, method: Literal["cubic", "forward", "centered"] = "cubic", **kwargs: Any
    ) -> np.ndarray:
        """Returns velocities in m/s.

        Args:
            method: The method to use for computing the derivatives.
            **kwargs: Additional arguments to pass to the cubic spline
                approximation.
        """
        return self._derivative(
            self.poses.vec3, self.timestamps_us, deriv=1, method=method, **kwargs
        )

    def accelerations(
        self, method: Literal["cubic", "forward", "centered"] = "cubic", **kwargs: Any
    ) -> np.ndarray:
        """Returns accelerations in m/s^2.

        Args:
            method: The method to use for computing the derivatives.
            **kwargs: Additional arguments to pass to the cubic spline
                approximation.
        """
        return self._derivative(
            self.poses.vec3, self.timestamps_us, deriv=2, method=method, **kwargs
        )

    def jerk(
        self, method: Literal["cubic", "forward", "centered"] = "cubic", **kwargs: Any
    ) -> np.ndarray:
        """Returns jerk in m/s^3.

        Args:
            method: The method to use for computing the derivatives.
            **kwargs: Additional arguments to pass to the cubic spline
                approximation.
        """
        return self._derivative(
            self.poses.vec3, self.timestamps_us, deriv=3, method=method, **kwargs
        )

    def yaw_rates(
        self, method: Literal["cubic", "forward", "centered"] = "cubic", **kwargs: Any
    ) -> np.ndarray:
        """Returns yaw rates in rad/s.

        Args:
            method: The method to use for computing the derivatives.
            **kwargs: Additional arguments to pass to the cubic spline
                approximation.
        """
        return self._derivative(
            np.unwrap(self.poses.yaw),
            self.timestamps_us,
            deriv=1,
            method=method,
            **kwargs,
        )

    def yaw_accelerations(
        self, method: Literal["cubic", "forward", "centered"] = "cubic", **kwargs: Any
    ) -> np.ndarray:
        """Returns yaw accelerations in rad/s^2.

        Args:
            method: The method to use for computing the derivatives.
            **kwargs: Additional arguments to pass to the cubic spline
                approximation.
        """
        return self._derivative(
            np.unwrap(self.poses.yaw),
            self.timestamps_us,
            deriv=2,
            method=method,
            **kwargs,
        )

    def append(self, other: Trajectory) -> Trajectory:
        """Append another trajectory to the end of `self`.

        Assumes that the two trajectories are continuous and that either they
        have one overlapping timestamp or that `self` ends before `other` starts.
        """
        if self.is_empty():
            return other.clone()
        if other.is_empty():
            return self.clone()
        if self.timestamps_us[-1] == other.timestamps_us[0]:
            if not (
                np.allclose(self.poses[-1].vec3, other.poses[0].vec3)
                and np.allclose(self.poses[-1].quat, other.poses[0].quat)
            ):
                raise ValueError(
                    "If both trajectories overlap by one timestamp, the last "
                    "pose of `self` must match the first pose of `other`"
                )
            return Trajectory(
                timestamps_us=np.concatenate(
                    [self.timestamps_us, other.timestamps_us[1:]]
                ),
                poses=self.poses.append(other.poses[1:]),
            )
        elif self.timestamps_us[-1] < other.timestamps_us[0]:
            return Trajectory(
                timestamps_us=np.concatenate([self.timestamps_us, other.timestamps_us]),
                poses=self.poses.append(other.poses),
            )
        else:
            raise ValueError("Trajectories are not continuous")

    @property
    def xyzh(self) -> np.ndarray:
        """Returns the x, y, z, and h values of the trajectory.

        Helpful for interoperability with trajdata which uses this format often.

        """
        return np.concatenate([self.poses.vec3, self.poses.yaw[..., None]], axis=-1)

    def to_polyline(self) -> Polyline:
        """Extract spatial path from the trajectory and drop timing information."""

        return Polyline(points=self.poses.vec3.copy())
