# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Type stubs for the utils_rs Rust extension module."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

def version() -> str:
    """Returns the version of this Rust extension."""
    ...

class Pose:
    """
    A single rigid transform: position + quaternion rotation.

    This is the atomic unit for pose representation. Quaternions are stored
    internally in scipy format (x, y, z, w) for compatibility with
    scipy.spatial.transform.Rotation.

    :param position: numpy array of shape (3,) with [x, y, z] position (float32 or float64)
    :param quaternion: numpy array of shape (4,) with [x, y, z, w] quaternion in scipy format (float32 or float64).
        Must be approximately unit-length; quaternions with small drift are silently
        normalized, while far-from-unit or zero quaternions raise ValueError.
    """

    def __init__(
        self,
        position: NDArray[np.floating],
        quaternion: NDArray[np.floating],
    ) -> None: ...
    @staticmethod
    def identity() -> Pose:
        """Create an identity pose (zero position, identity rotation)."""
        ...

    @staticmethod
    def from_denormalized_quat(
        position: NDArray[np.floating],
        quaternion: NDArray[np.floating],
    ) -> Pose:
        """
        Create a pose from a possibly non-unit quaternion by normalizing it.

        Use when the quaternion may not be unit-length (e.g. from an optimizer).
        Zero quaternions raise ValueError; any other quaternion is normalized.
        Quaternion is in scipy format [x, y, z, w].
        """
        ...

    @staticmethod
    def from_proto(
        position: NDArray[np.floating],
        quat_wxyz: NDArray[np.floating],
    ) -> Pose:
        """
        Create from proto format.

        :param position: numpy array of shape (3,) with [x, y, z] position (float32 or float64)
        :param quat_wxyz: numpy array of shape (4,) with [w, x, y, z] quaternion in proto order (float32 or float64).
            Must be approximately unit-length; small drift is silently normalized.
        """
        ...

    @staticmethod
    def from_se3(mat: NDArray[np.floating]) -> Pose:
        """
        Create from SE3 4x4 matrix.

        :param mat: 4x4 homogeneous transformation matrix (row-major)
        """
        ...

    @property
    def vec3(self) -> NDArray[np.float32]:
        """Position as numpy array of shape (3,)."""
        ...

    @property
    def quat(self) -> NDArray[np.float32]:
        """Quaternion as numpy array of shape (4,) in scipy format [x, y, z, w]."""
        ...

    def __matmul__(self, other: Pose) -> Pose:
        """
        Compose two poses: self @ other.

        Result: new_position = self.rotation * other.position + self.position
                new_rotation = self.rotation * other.rotation
        """
        ...

    def inverse(self) -> Pose:
        """
        Compute the inverse of this pose.

        If self transforms A -> B, then inverse transforms B -> A.
        """
        ...

    def yaw(self) -> float:
        """Extract yaw angle (rotation around Z axis) in radians."""
        ...

    def to_proto(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """
        Convert to proto format: ([x, y, z], [w, x, y, z]).

        Returns position and quaternion in gRPC proto order.
        Note: Proto quaternion order is (w, x, y, z), different from scipy.
        """
        ...

    def as_se3(self) -> NDArray[np.float32]:
        """Convert to SE3 4x4 matrix as numpy array."""
        ...

    def clone(self) -> Pose:
        """Create a copy of this pose."""
        ...

    def __eq__(self, other: object) -> bool:
        """Check equality with another pose (approximate, for floating point)."""
        ...

    def __repr__(self) -> str:
        """String representation for debugging."""
        ...

    def __hash__(self) -> int:
        """Hash for use in sets/dicts (based on position and quaternion values)."""
        ...

class Trajectory:
    """
    A trajectory of timestamped poses.

    This is the main trajectory type, providing efficient storage and operations
    for interpolation, transforms, derivatives, and incremental updates.

    :param timestamps: 1D array of timestamps in microseconds (must be strictly increasing, accepts int32/int64/uint32/uint64)
    :param positions: 2D array of shape (N, 3) with positions (float32 or float64)
    :param quaternions: 2D array of shape (N, 4) with quaternions (x, y, z, w) (float32 or float64).
        Each quaternion must be approximately unit-length; small drift is silently
        normalized, while far-from-unit or zero quaternions raise ValueError.
    """

    def __init__(
        self,
        timestamps: NDArray[np.integer],
        positions: NDArray[np.floating],
        quaternions: NDArray[np.floating],
    ) -> None: ...
    @staticmethod
    def create_empty() -> Trajectory:
        """Create an empty Trajectory."""
        ...

    @staticmethod
    def from_poses(
        timestamps: NDArray[np.uint64],
        poses: list[Pose],
    ) -> Trajectory:
        """
        Create a Trajectory from timestamps and a list of Pose objects.

        :param timestamps: 1D array of uint64 timestamps in microseconds
        :param poses: List of Pose objects
        :return: A new Trajectory instance
        """
        ...

    def __len__(self) -> int:
        """Number of poses in the trajectory."""
        ...

    def __repr__(self) -> str:
        """String representation for debugging."""
        ...

    def is_empty(self) -> bool:
        """Check if the trajectory is empty."""
        ...

    @property
    def timestamps_us(self) -> NDArray[np.uint64]:
        """Timestamps in microseconds as numpy array."""
        ...

    @property
    def time_range_us(self) -> range:
        """Time range as Python range(start_us, end_us)."""
        ...

    @property
    def positions(self) -> NDArray[np.float32]:
        """Positions as 2D numpy array of shape (N, 3)."""
        ...

    @property
    def quaternions(self) -> NDArray[np.float32]:
        """Quaternions as 2D numpy array of shape (N, 4) in scipy format (x, y, z, w)."""
        ...

    def rotation_matrices(self) -> NDArray[np.float32]:
        """
        Rotation matrices as numpy array of shape (N, 3, 3).

        Each matrix is a 3x3 rotation matrix derived from the quaternion,
        in row-major order (compatible with numpy/scipy conventions).
        """
        ...

    @property
    def yaws(self) -> NDArray[np.float32]:
        """Yaw angles as 1D numpy array of shape (N,)."""
        ...

    @property
    def first_pose(self) -> Pose:
        """Get the first pose. Raises IndexError if empty."""
        ...

    @property
    def last_pose(self) -> Pose:
        """Get the last pose. Raises IndexError if empty."""
        ...

    def get_pose(self, idx: int) -> Pose:
        """Get a single Pose at the given index."""
        ...

    def set_pose(self, idx: int, pose: Pose) -> None:
        """
        Set the pose at the given index in-place.

        :param idx: Index (supports negative indexing)
        :param pose: The new Pose to set at that index
        """
        ...

    def get_time_range_tuple(self) -> tuple[int, int]:
        """Get the time range as (start_us, end_us) tuple. Returns (0, 0) if empty."""
        ...
    # =========================================================================
    # Derivative Methods
    # =========================================================================

    def velocities(self, method: str = "centered") -> NDArray[np.float32]:
        """
        Compute velocities in m/s using finite differences.

        :param method: "centered" or "forward". Default: "centered"
        :return: 2D numpy array of shape (N, 3) with velocity vectors
        """
        ...

    def accelerations(self, method: str = "centered") -> NDArray[np.float32]:
        """
        Compute accelerations in m/s^2 using finite differences.

        :param method: "centered" or "forward". Default: "centered"
        :return: 2D numpy array of shape (N, 3) with acceleration vectors
        """
        ...

    def jerk(self, method: str = "centered") -> NDArray[np.float32]:
        """
        Compute jerk in m/s^3 using finite differences.

        :param method: "centered" or "forward". Default: "centered"
        :return: 2D numpy array of shape (N, 3) with jerk vectors
        """
        ...

    def yaw_rates(self, method: str = "centered") -> NDArray[np.float32]:
        """
        Compute yaw rates in rad/s using finite differences.

        :param method: "centered" or "forward". Default: "centered"
        :return: 1D numpy array of shape (N,) with yaw rate values
        """
        ...

    def yaw_accelerations(self, method: str = "centered") -> NDArray[np.float32]:
        """
        Compute yaw accelerations in rad/s^2 using finite differences.

        :param method: "centered" or "forward". Default: "centered"
        :return: 1D numpy array of shape (N,) with yaw acceleration values
        """
        ...
    # =========================================================================
    # Mutation Methods
    # =========================================================================

    def update_absolute(self, timestamp: int, pose: Pose) -> None:
        """
        Append a new pose with absolute coordinates.

        :param timestamp: Timestamp in microseconds (must be > last timestamp)
        :param pose: The Pose to append
        """
        ...

    def update_relative(self, timestamp: int, delta_pose: Pose) -> None:
        """
        Append a new pose relative to the last pose.

        The new pose is computed as: new_pose = last_pose @ delta_pose

        :param timestamp: Timestamp in microseconds (must be > last timestamp)
        :param delta_pose: The delta Pose in local frame
        """
        ...
    # =========================================================================
    # Transform Methods
    # =========================================================================

    def transform(self, transform: Pose, is_relative: bool = False) -> Trajectory:
        """
        Transform all poses by a given pose.

        :param transform: The pose to transform by.
        :param is_relative: If true, applies right multiplication (pose @ transform),
                           otherwise left multiplication (transform @ pose). Default: false.
        :return: A new Trajectory with transformed poses.
        """
        ...

    def clip(self, start_us: int, end_us: int) -> Trajectory:
        """
        Clip the trajectory to a time range, interpolating at boundaries.

        :param start_us: Start timestamp (inclusive)
        :param end_us: End timestamp (exclusive)
        :return: A new Trajectory clipped to the specified range.
        """
        ...

    def append(self, other: Trajectory) -> Trajectory:
        """
        Append another trajectory to the end of this one.

        The trajectories must be continuous: either they have one overlapping
        timestamp (with matching pose) or self ends before other starts.
        """
        ...

    def filter(self, mask: NDArray[np.bool_]) -> Trajectory:
        """
        Filter trajectory by boolean mask.

        :param mask: 1D boolean array of length N.
        :return: A new Trajectory with only the poses where mask is True.
        """
        ...

    def slice(self, start: int, end: int) -> Trajectory:
        """Slice trajectory from start to end (exclusive)."""
        ...

    def concat(self, other: Trajectory) -> Trajectory:
        """Concatenate another trajectory to the end."""
        ...

    def clone_storage(self) -> Trajectory:
        """Create a deep copy of this trajectory."""
        ...

    def clone(self) -> Trajectory:
        """Create a deep copy of this trajectory."""
        ...

    def to_polyline(self) -> Polyline:
        """
        Extract the spatial path as a Polyline, dropping timing information.

        :return: A 3D Polyline containing only the positions from this trajectory.
        """
        ...

    def interpolate_pose(self, at_us: int) -> Pose:
        """
        Interpolate a single pose at the given timestamp.

        :param at_us: Timestamp in microseconds (must be within trajectory range)
        :return: Interpolated Pose at the given timestamp.
        """
        ...

    def interpolate_delta(self, start_us: int, end_us: int) -> Pose:
        """
        Compute the relative transform between two timestamps.

        :return: start_pose.inverse() @ end_pose
        """
        ...

    def interpolate(self, target_timestamps: NDArray[np.uint64]) -> Trajectory:
        """
        Interpolate poses at the given timestamps and return a new Trajectory.

        :param target_timestamps: 1D array of uint64 timestamps to interpolate at.
        :return: A new Trajectory with the interpolated poses at the given timestamps.
        """
        ...

    def interpolate_poses_list(
        self, target_timestamps: NDArray[np.uint64]
    ) -> list[Pose]:
        """
        Interpolate poses at multiple timestamps and return as list of Pose objects.
        """
        ...

class Polyline:
    """
    A spatial polyline as an ordered set of 2D or 3D waypoints.

    This class provides high-performance operations for point projection,
    interpolation, resampling, and transforms.

    :param points: 2D array of shape (N, D) where D is 2 or 3.
    """

    def __init__(self, points: NDArray[np.floating]) -> None: ...
    @staticmethod
    def create_empty(dimension: int = 3) -> Polyline:
        """Factory for an empty polyline in the requested dimension."""
        ...

    def __len__(self) -> int:
        """Return the number of waypoints in the polyline."""
        ...

    def __repr__(self) -> str:
        """Summarise the polyline for debugging/logging purposes."""
        ...
    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def is_empty(self) -> bool:
        """Whether the polyline contains zero waypoints."""
        ...

    @property
    def dimension(self) -> int:
        """Spatial dimensionality of the polyline (2 or 3)."""
        ...

    @property
    def waypoints(self) -> NDArray[np.float32]:
        """Reference to the underlying waypoint array (shape (N, D))."""
        ...

    @property
    def points(self) -> NDArray[np.float32]:
        """Get points as numpy array of shape (N, D). Alias for waypoints."""
        ...

    @property
    def total_length(self) -> float:
        """Total arc length of the polyline."""
        ...

    @property
    def segment_lengths(self) -> NDArray[np.float32]:
        """Euclidean distance between consecutive waypoints."""
        ...
    # =========================================================================
    # Methods
    # =========================================================================

    def arc_lengths(self) -> NDArray[np.float32]:
        """Cumulative arc lengths along the polyline."""
        ...

    def project_point(
        self,
        point: NDArray[np.floating],
    ) -> tuple[NDArray[np.float32], int, float]:
        """
        Orthogonally project a point onto the polyline segments.

        :param point: Point to project (1D array of length D)
        :return: (projected_point, segment_idx, distance_along_segment)
        """
        ...

    def project_points_batch(
        self,
        points: NDArray[np.floating],
    ) -> tuple[NDArray[np.float32], NDArray[np.uintp], NDArray[np.float32]]:
        """
        Project multiple points onto the polyline in batch.

        :param points: 2D array of shape (M, D) with points to project.
        :return: (projected_points, segment_indices, distances_along)
        """
        ...

    def positions_at(self, distances: NDArray[np.floating]) -> NDArray[np.float32]:
        """
        Interpolate positions at specific distances along the polyline.

        :param distances: 1D array of distances along the polyline.
        :return: 2D array of shape (M, D) with interpolated positions.
        """
        ...

    def remaining_from_point(
        self,
        point: NDArray[np.floating],
    ) -> tuple[Polyline, tuple[NDArray[np.float32], int, float]]:
        """
        Return the polyline remainder after projecting a point.

        :param point: Point to project (1D array of length D)
        :return: (remaining_polyline, (projected_point, segment_idx, distance_along))
        """
        ...

    def resample_from_point(
        self,
        start_point: NDArray[np.floating],
        spacing: float,
        n_points: int,
    ) -> Polyline:
        """
        Uniformly resample the remainder of the polyline after a start point.

        :param start_point: Point to project onto the polyline.
        :param spacing: Distance between samples.
        :param n_points: Maximum number of points to sample.

        :return: A new Polyline with resampled points.
        """
        ...

    def clip(
        self,
        start: int | None = None,
        end: int | None = None,
    ) -> Polyline:
        """Return a copy over the waypoint slice [start:end]."""
        ...

    def append(self, other: Polyline) -> Polyline:
        """Concatenate another polyline with matching dimensionality."""
        ...

    def downsample_with_min_distance(self, min_distance: float) -> None:
        """
        Downsample ensuring minimum distance between waypoints.

        Modifies the polyline in place.
        """
        ...

    def get_cumulative_distances_from_point(
        self,
        point: NDArray[np.floating],
    ) -> tuple[NDArray[np.float32], float]:
        """
        Cumulative distances along the remainder of the polyline from a projected point.

        :param point: Point to project (1D array of length D)
        :return: (cumulative_distances, distance_to_projection)
        """
        ...

    def zero_out_z(self) -> Polyline:
        """Return a new polyline with the z coordinate set to zero (3D only)."""
        ...

    def transform(self, transform_pose: Pose) -> Polyline:
        """Apply a rigid transform to the waypoints (3D only)."""
        ...

    def clone(self) -> Polyline:
        """Clone the polyline."""
        ...
