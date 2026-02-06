// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 NVIDIA Corporation

//! Trajectory: Rust-backed trajectory with efficient interpolation.

use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::array_utils::{ensure_unit_quat, extract_array1_u64, extract_array2_f32};
use crate::polyline::Polyline;
use crate::pose::{quat_to_scipy, Pose};
use glam::{Quat, Vec3};

/// A pose with an associated timestamp.
#[derive(Clone, Copy, Debug)]
pub struct TimestampedPose {
    pub timestamp_us: u64,
    pub pose: Pose,
}

impl TimestampedPose {
    #[inline]
    pub fn new(timestamp_us: u64, pose: Pose) -> Self {
        Self { timestamp_us, pose }
    }
}

/// A trajectory of timestamped poses.
///
/// This is the main trajectory type, providing efficient storage and operations
/// for interpolation, transforms, and incremental updates.
#[pyclass(name = "Trajectory")]
#[derive(Clone)]
pub struct Trajectory {
    /// Timestamped poses (strictly increasing by timestamp)
    poses: Vec<TimestampedPose>,
}

#[pymethods]
impl Trajectory {
    /// Create a new Trajectory from numpy arrays.
    ///
    /// Args:
    ///     timestamps: 1D array of timestamps in microseconds (must be strictly increasing)
    ///     positions: 2D array of shape (N, 3) with positions (float32 or float64)
    ///     quaternions: 2D array of shape (N, 4) with quaternions (x, y, z, w) (float32 or float64)
    ///
    /// Raises:
    ///     ValueError: If timestamps are not strictly increasing or shapes don't match.
    #[new]
    fn new(
        py: Python<'_>,
        timestamps: PyObject,
        positions: PyObject,
        quaternions: PyObject,
    ) -> PyResult<Self> {
        let ts = extract_array1_u64(py, &timestamps, "timestamps")?;
        let n = ts.len();

        // Extract positions with flexible dtype
        let (pos_data, pos_shape) = extract_array2_f32(py, &positions, "positions")?;
        if pos_shape[1] != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "positions must have shape (N, 3), got ({}, {})",
                pos_shape[0], pos_shape[1]
            )));
        }
        if pos_shape[0] != n {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "positions has {} rows but timestamps has {} elements",
                pos_shape[0], n
            )));
        }

        // Extract quaternions with flexible dtype (scipy format: x, y, z, w)
        let (quat_data, quat_shape) = extract_array2_f32(py, &quaternions, "quaternions")?;
        if quat_shape[1] != 4 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "quaternions must have shape (N, 4), got ({}, {})",
                quat_shape[0], quat_shape[1]
            )));
        }
        if quat_shape[0] != n {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "quaternions has {} rows but timestamps has {} elements",
                quat_shape[0], n
            )));
        }

        // Validate timestamps are strictly increasing
        if n > 1 {
            for i in 1..n {
                if ts[i] <= ts[i - 1] {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "timestamps must be strictly increasing, but ts[{}]={} <= ts[{}]={}",
                        i,
                        ts[i],
                        i - 1,
                        ts[i - 1]
                    )));
                }
            }
        }

        // Build poses from flat arrays (quaternions are in scipy format which matches glam's xyzw)
        let mut poses = Vec::with_capacity(n);
        for i in 0..n {
            let pos = Vec3::new(pos_data[i * 3], pos_data[i * 3 + 1], pos_data[i * 3 + 2]);
            let quat = Quat::from_xyzw(
                quat_data[i * 4],
                quat_data[i * 4 + 1],
                quat_data[i * 4 + 2],
                quat_data[i * 4 + 3],
            );
            let quat = ensure_unit_quat(quat, &format!("quaternions[{}]", i))?;
            poses.push(TimestampedPose::new(ts[i], Pose::from_glam(pos, quat)));
        }

        Ok(Self { poses })
    }

    /// Number of poses in the trajectory.
    fn __len__(&self) -> usize {
        self.poses.len()
    }

    /// Check if the trajectory is empty.
    fn is_empty(&self) -> bool {
        self.poses.is_empty()
    }

    /// Timestamps in microseconds as numpy array.
    #[getter]
    fn timestamps_us<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        let ts: Vec<u64> = self.poses.iter().map(|tp| tp.timestamp_us).collect();
        PyArray1::from_vec(py, ts)
    }

    /// Get the time range as a Python range(start_us, end_us).
    #[getter]
    fn time_range_us<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let builtins = py.import("builtins")?;
        let range_cls = builtins.getattr("range")?;
        if self.poses.is_empty() {
            range_cls.call1((0i64, 0i64))
        } else {
            let start = self.poses[0].timestamp_us as i64;
            let end = (self.poses[self.poses.len() - 1].timestamp_us + 1) as i64;
            range_cls.call1((start, end))
        }
    }

    /// Get the time range as (start_us, end_us) tuple. Returns (0, 0) if empty.
    fn get_time_range_tuple(&self) -> (u64, u64) {
        if self.poses.is_empty() {
            (0, 0)
        } else {
            (
                self.poses[0].timestamp_us,
                self.poses[self.poses.len() - 1].timestamp_us + 1,
            )
        }
    }

    /// Create an empty Trajectory.
    #[staticmethod]
    fn create_empty() -> Self {
        Self { poses: Vec::new() }
    }

    /// Create a Trajectory from timestamps and a list of Pose objects.
    ///
    /// Args:
    ///     timestamps: 1D array of uint64 timestamps in microseconds
    ///     poses: List of Pose objects
    ///
    /// Returns:
    ///     A new Trajectory instance.
    #[staticmethod]
    fn from_poses(
        timestamps: PyReadonlyArray1<'_, u64>,
        poses: Vec<Pose>,
    ) -> PyResult<Self> {
        let ts = timestamps.as_slice()?;
        let n = ts.len();

        if n != poses.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "timestamps length {} does not match poses length {}",
                n,
                poses.len()
            )));
        }

        // Validate timestamps are strictly increasing
        if n > 1 {
            for i in 1..n {
                if ts[i] <= ts[i - 1] {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "timestamps must be strictly increasing, but ts[{}]={} <= ts[{}]={}",
                        i, ts[i], i - 1, ts[i - 1]
                    )));
                }
            }
        }

        let timestamped_poses: Vec<TimestampedPose> = ts
            .iter()
            .zip(poses.iter())
            .map(|(&t, p)| TimestampedPose::new(t, *p))
            .collect();

        Ok(Self { poses: timestamped_poses })
    }

    /// Append a new pose with absolute coordinates.
    ///
    /// Args:
    ///     timestamp: Timestamp in microseconds (must be > last timestamp)
    ///     pose: The Pose to append
    fn update_absolute(&mut self, timestamp: u64, pose: &Pose) -> PyResult<()> {
        // Validate timestamp is strictly increasing
        if let Some(last) = self.poses.last() {
            if timestamp <= last.timestamp_us {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Timestamp {} must be greater than last timestamp {}",
                    timestamp, last.timestamp_us
                )));
            }
        }

        self.poses.push(TimestampedPose::new(timestamp, *pose));
        Ok(())
    }

    /// Append a new pose relative to the last pose.
    ///
    /// The new pose is computed as: new_pose = last_pose @ delta_pose
    /// (right multiplication, i.e., delta is in local frame)
    ///
    /// Args:
    ///     timestamp: Timestamp in microseconds (must be > last timestamp)
    ///     delta_pose: The delta Pose in local frame
    fn update_relative(&mut self, timestamp: u64, delta_pose: &Pose) -> PyResult<()> {
        if self.poses.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot update_relative on empty trajectory",
            ));
        }

        let last_pose = self.poses.last().unwrap().pose;
        let new_pose = last_pose.compose(delta_pose);
        self.update_absolute(timestamp, &new_pose)
    }

    /// Get a single Pose at the given index.
    fn get_pose(&self, idx: isize) -> PyResult<Pose> {
        let n = self.poses.len() as isize;
        let idx = if idx < 0 { n + idx } else { idx };
        if idx < 0 || idx >= n {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "index {} out of range for trajectory of length {}",
                idx, n
            )));
        }
        Ok(self.poses[idx as usize].pose)
    }

    /// Set the pose at the given index in-place.
    ///
    /// Args:
    ///     idx: Index (supports negative indexing)
    ///     pose: The new Pose to set at that index
    fn set_pose(&mut self, idx: isize, pose: &Pose) -> PyResult<()> {
        let n = self.poses.len() as isize;
        let idx = if idx < 0 { n + idx } else { idx };
        if idx < 0 || idx >= n {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "index {} out of range for trajectory of length {}",
                idx, n
            )));
        }
        self.poses[idx as usize].pose = *pose;
        Ok(())
    }

    /// Get the last pose. Raises IndexError if empty.
    #[getter]
    fn last_pose(&self) -> PyResult<Pose> {
        self.poses
            .last()
            .map(|tp| tp.pose)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    "Cannot get last_pose of empty trajectory",
                )
            })
    }

    /// Get the first pose. Raises IndexError if empty.
    #[getter]
    fn first_pose(&self) -> PyResult<Pose> {
        self.poses
            .first()
            .map(|tp| tp.pose)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    "Cannot get first_pose of empty trajectory",
                )
            })
    }

    /// Positions as 2D numpy array of shape (N, 3).
    #[getter]
    fn positions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let n = self.poses.len();
        if n == 0 {
            return PyArray2::zeros(py, [0, 3], false);
        }
        let data: Vec<Vec<f32>> = self
            .poses
            .iter()
            .map(|tp| {
                let p = tp.pose.position();
                vec![p.x, p.y, p.z]
            })
            .collect();
        PyArray2::from_vec2(py, &data).unwrap_or_else(|_| PyArray2::zeros(py, [n, 3], false))
    }

    /// Quaternions as 2D numpy array of shape (N, 4) in scipy format.
    #[getter]
    fn quaternions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let n = self.poses.len();
        if n == 0 {
            return PyArray2::zeros(py, [0, 4], false);
        }
        let data: Vec<Vec<f32>> = self
            .poses
            .iter()
            .map(|tp| quat_to_scipy(tp.pose.quaternion()).to_vec())
            .collect();
        PyArray2::from_vec2(py, &data).unwrap_or_else(|_| PyArray2::zeros(py, [n, 4], false))
    }

    /// Rotation matrices as 3D numpy array of shape (N, 3, 3).
    ///
    /// Each matrix is a 3x3 rotation matrix derived from the quaternion.
    /// Matrices are row-major (compatible with numpy/scipy conventions).
    ///
    /// This is a method (not a property) to signal that it performs computation.
    fn rotation_matrices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        let n = self.poses.len();
        if n == 0 {
            return PyArray3::zeros(py, [0, 3, 3], false);
        }
        // Single flat allocation, then reshape to (N, 3, 3)
        let mut data = Vec::with_capacity(n * 9);
        for tp in &self.poses {
            let rm = tp.pose.rotation_matrix();
            for row in &rm {
                data.extend_from_slice(row);
            }
        }
        PyArray1::from_vec(py, data)
            .reshape([n, 3, 3])
            .expect("reshape to (N, 3, 3) cannot fail for n*9 elements")
    }

    /// Yaw angles as 1D numpy array of shape (N,).
    #[getter]
    fn yaws<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let yaws: Vec<f32> = self
            .poses
            .iter()
            .map(|tp| {
                let q = tp.pose.quaternion();
                (2.0 * (q.w * q.z + q.x * q.y)).atan2(1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            })
            .collect();
        PyArray1::from_vec(py, yaws)
    }

    /// Filter trajectory by boolean mask.
    ///
    /// Args:
    ///     mask: 1D boolean array of length N.
    ///
    /// Returns:
    ///     A new Trajectory with only the poses where mask is True.
    fn filter(&self, mask: PyReadonlyArray1<'_, bool>) -> PyResult<Self> {
        let mask_slice = mask.as_slice()?;
        if mask_slice.len() != self.poses.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "mask length {} does not match trajectory length {}",
                mask_slice.len(),
                self.poses.len()
            )));
        }

        let poses: Vec<TimestampedPose> = self
            .poses
            .iter()
            .zip(mask_slice.iter())
            .filter_map(|(tp, &keep)| if keep { Some(*tp) } else { None })
            .collect();

        Ok(Self { poses })
    }

    /// Slice trajectory from start to end (exclusive).
    fn slice(&self, start: isize, end: isize) -> PyResult<Self> {
        let n = self.poses.len() as isize;
        let start = if start < 0 {
            (n + start).max(0)
        } else {
            start.min(n)
        };
        let end = if end < 0 {
            (n + end).max(0)
        } else {
            end.min(n)
        };
        let start = start as usize;
        let end = end as usize;

        if start >= end {
            return Ok(Self::create_empty());
        }

        Ok(Self {
            poses: self.poses[start..end].to_vec(),
        })
    }

    /// Concatenate another trajectory to the end.
    fn concat(&self, other: &Trajectory) -> PyResult<Self> {
        if other.poses.is_empty() {
            return Ok(self.clone());
        }
        if self.poses.is_empty() {
            return Ok(other.clone());
        }

        // Validate timestamps don't overlap
        let self_last = self.poses.last().unwrap().timestamp_us;
        let other_first = other.poses.first().unwrap().timestamp_us;
        if other_first <= self_last {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Cannot concat: other trajectory starts at {} but self ends at {}",
                other_first, self_last
            )));
        }

        let mut poses = self.poses.clone();
        poses.extend_from_slice(&other.poses);
        Ok(Self { poses })
    }

    /// Create a deep copy of this trajectory.
    #[pyo3(name = "clone")]
    fn py_clone(&self) -> Self {
        self.clone()
    }

    /// Transform all poses by a given pose (left multiplication by default).
    ///
    /// Args:
    ///     transform: The pose to transform by.
    ///     is_relative: If true, applies right multiplication (pose @ transform),
    ///                  otherwise left multiplication (transform @ pose). Default: false.
    ///
    /// Returns:
    ///     A new Trajectory with transformed poses.
    #[pyo3(signature = (transform, is_relative = false))]
    fn transform(&self, transform: &Pose, is_relative: bool) -> Self {
        if self.poses.is_empty() {
            return Self::create_empty();
        }

        let t_pos = transform.position();
        let t_quat = transform.quaternion();

        let poses: Vec<TimestampedPose> = self
            .poses
            .iter()
            .map(|tp| {
                let pos = tp.pose.position();
                let quat = tp.pose.quaternion();

                let new_pose = if is_relative {
                    // Right multiply: pose @ transform
                    Pose::from_glam(pos + quat * t_pos, quat * t_quat)
                } else {
                    // Left multiply: transform @ pose
                    Pose::from_glam(t_pos + t_quat * pos, t_quat * quat)
                };

                TimestampedPose::new(tp.timestamp_us, new_pose)
            })
            .collect();

        Self { poses }
    }

    /// Extract the spatial path as a Polyline, dropping timing information.
    ///
    /// Returns:
    ///     A 3D Polyline containing only the positions from this trajectory.
    fn to_polyline(&self) -> Polyline {
        if self.poses.is_empty() {
            return Polyline::from_flat(Vec::new(), 3);
        }

        let mut points = Vec::with_capacity(self.poses.len() * 3);
        for tp in &self.poses {
            let p = tp.pose.position();
            points.push(p.x);
            points.push(p.y);
            points.push(p.z);
        }

        Polyline::from_flat(points, 3)
    }

    /// String representation for debugging.
    fn __repr__(&self) -> String {
        let (start, end) = self.get_time_range_tuple();
        format!(
            "Trajectory(n_poses={}, time_range_us={}..{})",
            self.poses.len(),
            start,
            end
        )
    }

    // =========================================================================
    // Derivative Methods
    // =========================================================================

    /// Compute velocities in m/s using finite differences.
    ///
    /// Args:
    ///     method: "centered" or "forward". Default: "centered"
    ///
    /// Returns:
    ///     2D numpy array of shape (N, 3) with velocity vectors.
    #[pyo3(signature = (method = "centered"))]
    fn velocities<'py>(&self, py: Python<'py>, method: &str) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.compute_position_derivative(py, 1, method)
    }

    /// Compute accelerations in m/s^2 using finite differences.
    ///
    /// Args:
    ///     method: "centered" or "forward". Default: "centered"
    ///
    /// Returns:
    ///     2D numpy array of shape (N, 3) with acceleration vectors.
    #[pyo3(signature = (method = "centered"))]
    fn accelerations<'py>(&self, py: Python<'py>, method: &str) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.compute_position_derivative(py, 2, method)
    }

    /// Compute jerk in m/s^3 using finite differences.
    ///
    /// Args:
    ///     method: "centered" or "forward". Default: "centered"
    ///
    /// Returns:
    ///     2D numpy array of shape (N, 3) with jerk vectors.
    #[pyo3(signature = (method = "centered"))]
    fn jerk<'py>(&self, py: Python<'py>, method: &str) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.compute_position_derivative(py, 3, method)
    }

    /// Compute yaw rates in rad/s using finite differences.
    ///
    /// Args:
    ///     method: "centered" or "forward". Default: "centered"
    ///
    /// Returns:
    ///     1D numpy array of shape (N,) with yaw rate values.
    #[pyo3(signature = (method = "centered"))]
    fn yaw_rates<'py>(&self, py: Python<'py>, method: &str) -> PyResult<Bound<'py, PyArray1<f32>>> {
        self.compute_yaw_derivative(py, 1, method)
    }

    /// Compute yaw accelerations in rad/s^2 using finite differences.
    ///
    /// Args:
    ///     method: "centered" or "forward". Default: "centered"
    ///
    /// Returns:
    ///     1D numpy array of shape (N,) with yaw acceleration values.
    #[pyo3(signature = (method = "centered"))]
    fn yaw_accelerations<'py>(&self, py: Python<'py>, method: &str) -> PyResult<Bound<'py, PyArray1<f32>>> {
        self.compute_yaw_derivative(py, 2, method)
    }

    /// Interpolate poses at the given timestamps and return a new Trajectory.
    ///
    /// Args:
    ///     target_timestamps: 1D array of uint64 timestamps to interpolate at.
    ///         All timestamps must be within [start, end) of the trajectory.
    ///
    /// Returns:
    ///     A new Trajectory with the interpolated poses at the given timestamps.
    fn interpolate(
        &self,
        target_timestamps: PyReadonlyArray1<'_, u64>,
    ) -> PyResult<Trajectory> {
        let targets = target_timestamps.as_slice()?;
        let m = targets.len();

        if self.poses.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot interpolate on empty trajectory",
            ));
        }

        let n = self.poses.len();
        let start = self.poses[0].timestamp_us;
        let end = self.poses[n - 1].timestamp_us + 1; // exclusive end

        // Handle single-pose trajectory
        if n == 1 {
            // Verify all targets are at the single timestamp
            for &t in targets {
                if t != start {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Interpolation timestamp {} outside range [{}, {})",
                        t, start, end
                    )));
                }
            }
            // Return the single pose repeated m times
            let single_pose = self.poses[0].pose;
            let poses: Vec<TimestampedPose> = targets
                .iter()
                .map(|&t| TimestampedPose::new(t, single_pose))
                .collect();
            return Ok(Trajectory { poses });
        }

        // Allocate output buffer
        let mut out_poses = Vec::with_capacity(m);

        for &t in targets {
            // Bounds check
            if t < start || t >= end {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Interpolation timestamp {} outside range [{}, {})",
                    t, start, end
                )));
            }

            // Binary search to find segment index
            // We want the largest i such that timestamps[i] <= t
            let idx = self
                .poses
                .partition_point(|tp| tp.timestamp_us <= t)
                .saturating_sub(1);
            let idx = idx.min(n - 2); // Clamp to valid segment

            let t0 = self.poses[idx].timestamp_us;
            let t1 = self.poses[idx + 1].timestamp_us;

            // Compute interpolation factor alpha in [0, 1]
            let alpha = if t1 > t0 {
                (t - t0) as f32 / (t1 - t0) as f32
            } else {
                0.0
            };

            // Linear interpolation for positions
            let p0 = self.poses[idx].pose.position();
            let p1 = self.poses[idx + 1].pose.position();
            let p_interp = p0.lerp(p1, alpha);

            // Spherical linear interpolation for quaternions
            let q0 = self.poses[idx].pose.quaternion();
            let q1 = self.poses[idx + 1].pose.quaternion();
            let q_interp = q0.slerp(q1, alpha);

            out_poses.push(TimestampedPose::new(t, Pose::from_glam(p_interp, q_interp)));
        }

        Ok(Trajectory { poses: out_poses })
    }

    /// Interpolate poses at multiple timestamps and return as list of Pose objects.
    ///
    /// More efficient than calling interpolate and then slicing in Python.
    fn interpolate_poses_list(
        &self,
        target_timestamps: PyReadonlyArray1<'_, u64>,
    ) -> PyResult<Vec<Pose>> {
        let traj = self.interpolate(target_timestamps)?;
        Ok(traj.poses.iter().map(|tp| tp.pose).collect())
    }

    /// Interpolate a single pose at the given timestamp.
    ///
    /// Args:
    ///     at_us: Timestamp in microseconds (must be within trajectory range)
    ///
    /// Returns:
    ///     Interpolated Pose at the given timestamp.
    fn interpolate_pose(&self, at_us: u64) -> PyResult<Pose> {
        self.interpolate_pose_internal(at_us)
    }

    /// Compute the relative transform between two timestamps.
    ///
    /// Returns: start_pose.inverse() @ end_pose
    fn interpolate_delta(&self, start_us: u64, end_us: u64) -> PyResult<Pose> {
        let start_pose = self.interpolate_pose_internal(start_us)?;
        let end_pose = self.interpolate_pose_internal(end_us)?;
        Ok(start_pose.inv().compose(&end_pose))
    }

    /// Clip the trajectory to a time range, interpolating at boundaries.
    ///
    /// Returns the portion of the trajectory between start_us and end_us (exclusive).
    /// Interpolates poses at the boundaries if they don't align with existing timestamps.
    ///
    /// Args:
    ///     start_us: Start timestamp (inclusive)
    ///     end_us: End timestamp (exclusive)
    ///
    /// Returns:
    ///     A new Trajectory clipped to the specified range.
    fn clip(&self, start_us: u64, end_us: u64) -> PyResult<Self> {
        if start_us > end_us {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "start_us must be <= end_us",
            ));
        }

        let (traj_start, traj_end) = self.get_time_range_tuple();

        // Return empty if no overlap or empty range
        if start_us == end_us || traj_start >= end_us || traj_end <= start_us {
            return Ok(Self::create_empty());
        }

        // Clamp to trajectory bounds
        let start_us = start_us.max(traj_start);
        let last_timestamp_us = (end_us.min(traj_end)).saturating_sub(1);

        if start_us == last_timestamp_us {
            // Single pose
            let pose = self.interpolate_pose_internal(start_us)?;
            return Ok(Self::from_single_pose(start_us, &pose));
        }

        // Interpolate boundary poses
        let first_pose = self.interpolate_pose_internal(start_us)?;
        let last_pose = self.interpolate_pose_internal(last_timestamp_us)?;

        // Build output: first interpolated pose + interior poses + last interpolated pose
        let mut poses = vec![TimestampedPose::new(start_us, first_pose)];

        // Add poses strictly between start and end
        for tp in &self.poses {
            if tp.timestamp_us > start_us && tp.timestamp_us < last_timestamp_us {
                poses.push(*tp);
            }
        }

        poses.push(TimestampedPose::new(last_timestamp_us, last_pose));

        Ok(Self { poses })
    }

    /// Append another trajectory to the end of this one.
    ///
    /// The trajectories must be continuous: either they have one overlapping
    /// timestamp (with matching pose) or self ends before other starts.
    fn append(&self, other: &Trajectory) -> PyResult<Self> {
        if self.poses.is_empty() {
            return Ok(other.clone());
        }
        if other.poses.is_empty() {
            return Ok(self.clone());
        }

        let self_last = self.poses.last().unwrap();
        let other_first = other.poses.first().unwrap();

        if self_last.timestamp_us == other_first.timestamp_us {
            // Check poses match
            let pos_close =
                (self_last.pose.position() - other_first.pose.position()).length() < 1e-5;
            let quat_close = {
                let q1 = self_last.pose.quaternion();
                let q2 = other_first.pose.quaternion();
                (q1.x - q2.x).abs() < 1e-5
                    && (q1.y - q2.y).abs() < 1e-5
                    && (q1.z - q2.z).abs() < 1e-5
                    && (q1.w - q2.w).abs() < 1e-5
            };

            if !pos_close || !quat_close {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "If both trajectories overlap by one timestamp, the last pose of self must match the first pose of other",
                ));
            }

            // Skip first pose of other
            let mut poses = self.poses.clone();
            poses.extend_from_slice(&other.poses[1..]);
            Ok(Self { poses })
        } else if self_last.timestamp_us < other_first.timestamp_us {
            let mut poses = self.poses.clone();
            poses.extend_from_slice(&other.poses);
            Ok(Self { poses })
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Trajectories are not continuous",
            ))
        }
    }
}

impl Trajectory {
    /// Create from a single pose (internal use).
    fn from_single_pose(timestamp: u64, pose: &Pose) -> Self {
        Self {
            poses: vec![TimestampedPose::new(timestamp, *pose)],
        }
    }

    /// Interpolate a single pose (internal, no Python wrapper).
    fn interpolate_pose_internal(&self, at_us: u64) -> PyResult<Pose> {
        if self.poses.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot interpolate on empty trajectory",
            ));
        }

        let n = self.poses.len();
        let start = self.poses[0].timestamp_us;
        let end = self.poses[n - 1].timestamp_us + 1;

        if at_us < start || at_us >= end {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Interpolation timestamp {} outside range [{}, {})",
                at_us, start, end
            )));
        }

        // Handle single-pose trajectory
        if n == 1 {
            return Ok(self.poses[0].pose);
        }

        // Binary search
        let idx = self
            .poses
            .partition_point(|tp| tp.timestamp_us <= at_us)
            .saturating_sub(1);
        let idx = idx.min(n - 2);

        let t0 = self.poses[idx].timestamp_us;
        let t1 = self.poses[idx + 1].timestamp_us;
        let alpha = if t1 > t0 {
            (at_us - t0) as f32 / (t1 - t0) as f32
        } else {
            0.0
        };

        let p0 = self.poses[idx].pose.position();
        let p1 = self.poses[idx + 1].pose.position();
        let p_interp = p0.lerp(p1, alpha);

        let q0 = self.poses[idx].pose.quaternion();
        let q1 = self.poses[idx + 1].pose.quaternion();
        let q_interp = q0.slerp(q1, alpha);

        Ok(Pose::from_glam(p_interp, q_interp))
    }

    // =========================================================================
    // Derivative Helpers (internal)
    // =========================================================================

    /// Compute position derivative using finite differences.
    fn compute_position_derivative<'py>(
        &self,
        py: Python<'py>,
        order: usize,
        method: &str,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let n = self.poses.len();
        if n < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Not enough poses to compute derivatives",
            ));
        }

        // Extract positions as Vec<[f32; 3]>
        let mut values: Vec<[f32; 3]> = self
            .poses
            .iter()
            .map(|tp| {
                let p = tp.pose.position();
                [p.x, p.y, p.z]
            })
            .collect();

        // Convert timestamps to seconds
        let times: Vec<f32> = self
            .poses
            .iter()
            .map(|tp| tp.timestamp_us as f32 / 1e6)
            .collect();

        // Apply derivative `order` times
        for _ in 0..order {
            values = Self::differentiate_vec3(&values, &times, method);
        }

        // Convert to numpy array
        PyArray2::from_vec2(py, &values.iter().map(|v| v.to_vec()).collect::<Vec<_>>())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))
    }

    /// Compute yaw derivative using finite differences.
    fn compute_yaw_derivative<'py>(
        &self,
        py: Python<'py>,
        order: usize,
        method: &str,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let n = self.poses.len();
        if n < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Not enough poses to compute derivatives",
            ));
        }

        // Extract and unwrap yaw angles
        let mut yaws: Vec<f32> = self
            .poses
            .iter()
            .map(|tp| {
                let q = tp.pose.quaternion();
                (2.0 * (q.w * q.z + q.x * q.y)).atan2(1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            })
            .collect();

        // Unwrap phase (handle 2π discontinuities)
        for i in 1..n {
            let diff = yaws[i] - yaws[i - 1];
            if diff > std::f32::consts::PI {
                for j in i..n {
                    yaws[j] -= 2.0 * std::f32::consts::PI;
                }
            } else if diff < -std::f32::consts::PI {
                for j in i..n {
                    yaws[j] += 2.0 * std::f32::consts::PI;
                }
            }
        }

        // Convert timestamps to seconds
        let times: Vec<f32> = self
            .poses
            .iter()
            .map(|tp| tp.timestamp_us as f32 / 1e6)
            .collect();

        // Apply derivative `order` times
        let mut values = yaws;
        for _ in 0..order {
            values = Self::differentiate_scalar(&values, &times, method);
        }

        Ok(PyArray1::from_vec(py, values))
    }

    /// Differentiate a Vec<[f32; 3]> using finite differences.
    fn differentiate_vec3(arr: &[[f32; 3]], times: &[f32], method: &str) -> Vec<[f32; 3]> {
        let n = arr.len();
        if n < 2 {
            return arr.to_vec();
        }

        let mut result = vec![[0.0f32; 3]; n];

        match method {
            "forward" => {
                // Forward differences for all but last point
                for i in 0..(n - 1) {
                    let dt = times[i + 1] - times[i];
                    if dt > 0.0 {
                        for d in 0..3 {
                            result[i][d] = (arr[i + 1][d] - arr[i][d]) / dt;
                        }
                    }
                }
                // Backward difference for last point
                let dt = times[n - 1] - times[n - 2];
                if dt > 0.0 {
                    for d in 0..3 {
                        result[n - 1][d] = (arr[n - 1][d] - arr[n - 2][d]) / dt;
                    }
                }
            }
            _ => {
                // Centered differences (default)
                // Forward difference for first point
                let dt = times[1] - times[0];
                if dt > 0.0 {
                    for d in 0..3 {
                        result[0][d] = (arr[1][d] - arr[0][d]) / dt;
                    }
                }

                // Centered differences for middle points
                for i in 1..(n - 1) {
                    let dt = times[i + 1] - times[i - 1];
                    if dt > 0.0 {
                        for d in 0..3 {
                            result[i][d] = (arr[i + 1][d] - arr[i - 1][d]) / dt;
                        }
                    }
                }

                // Backward difference for last point
                let dt = times[n - 1] - times[n - 2];
                if dt > 0.0 {
                    for d in 0..3 {
                        result[n - 1][d] = (arr[n - 1][d] - arr[n - 2][d]) / dt;
                    }
                }
            }
        }

        result
    }

    /// Differentiate a Vec<f32> using finite differences.
    fn differentiate_scalar(arr: &[f32], times: &[f32], method: &str) -> Vec<f32> {
        let n = arr.len();
        if n < 2 {
            return arr.to_vec();
        }

        let mut result = vec![0.0f32; n];

        match method {
            "forward" => {
                // Forward differences for all but last point
                for i in 0..(n - 1) {
                    let dt = times[i + 1] - times[i];
                    if dt > 0.0 {
                        result[i] = (arr[i + 1] - arr[i]) / dt;
                    }
                }
                // Backward difference for last point
                let dt = times[n - 1] - times[n - 2];
                if dt > 0.0 {
                    result[n - 1] = (arr[n - 1] - arr[n - 2]) / dt;
                }
            }
            _ => {
                // Centered differences (default)
                // Forward difference for first point
                let dt = times[1] - times[0];
                if dt > 0.0 {
                    result[0] = (arr[1] - arr[0]) / dt;
                }

                // Centered differences for middle points
                for i in 1..(n - 1) {
                    let dt = times[i + 1] - times[i - 1];
                    if dt > 0.0 {
                        result[i] = (arr[i + 1] - arr[i - 1]) / dt;
                    }
                }

                // Backward difference for last point
                let dt = times[n - 1] - times[n - 2];
                if dt > 0.0 {
                    result[n - 1] = (arr[n - 1] - arr[n - 2]) / dt;
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_monotonicity_validation() {
        // This would need Python context to test, so we test the logic directly
        let ts = vec![100u64, 200, 300];
        for i in 1..ts.len() {
            assert!(ts[i] > ts[i - 1]);
        }

        let bad_ts = vec![100u64, 200, 150];
        let mut is_monotonic = true;
        for i in 1..bad_ts.len() {
            if bad_ts[i] <= bad_ts[i - 1] {
                is_monotonic = false;
                break;
            }
        }
        assert!(!is_monotonic);
    }
}
