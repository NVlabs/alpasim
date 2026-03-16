// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 NVIDIA Corporation

//! DynamicTrajectory: Trajectory with per-pose dynamic state.
//!
//! Internally delegates all pose operations to a contained [`Trajectory`],
//! keeping a parallel `Vec<[f64; 12]>` for dynamics (linear velocity,
//! angular velocity, linear acceleration, angular acceleration).

use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::prelude::*;

use crate::pose::{quat_to_scipy, Pose};
use crate::trajectory::Trajectory;

/// A trajectory of timestamped poses with per-pose dynamic states.
///
/// Wraps a [`Trajectory`] with parallel dynamics data (4 Vec3 fields = 12
/// floats per pose). Delegates all pose-level operations to the inner
/// trajectory and only manages the dynamics array itself.
#[pyclass(name = "DynamicTrajectory")]
#[derive(Clone)]
pub struct DynamicTrajectory {
    trajectory: Trajectory,
    /// Per-pose dynamics: [lin_vel(3), ang_vel(3), lin_accel(3), ang_accel(3)]
    dynamics: Vec<[f64; 12]>,
}

/// Extract an (N, 12) dynamics array from a numpy PyReadonlyArray2, validating
/// that its row count matches `expected_rows`.
fn extract_dynamics(
    dynamics: &PyReadonlyArray2<'_, f64>,
    expected_rows: usize,
    row_label: &str,
) -> PyResult<Vec<[f64; 12]>> {
    let shape = dynamics.shape();
    if shape[0] != expected_rows {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "dynamics has {} rows but {} has {} elements",
            shape[0], row_label, expected_rows
        )));
    }
    if shape[1] != 12 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "dynamics must have shape (N, 12), got ({}, {})",
            shape[0], shape[1]
        )));
    }

    let view = dynamics.as_array();
    let mut out = Vec::with_capacity(expected_rows);
    for i in 0..expected_rows {
        let mut row = [0.0f64; 12];
        for j in 0..12 {
            row[j] = view[[i, j]];
        }
        out.push(row);
    }
    Ok(out)
}

#[pymethods]
impl DynamicTrajectory {
    /// Create a new DynamicTrajectory from numpy arrays.
    ///
    /// Args:
    ///     timestamps: 1D uint64 array of timestamps in microseconds (strictly increasing)
    ///     positions: (N, 3) f32/f64 array
    ///     quaternions: (N, 4) f32/f64 array in scipy format (x, y, z, w)
    ///     dynamics: (N, 12) f64 array
    #[new]
    fn new(
        py: Python<'_>,
        timestamps: PyObject,
        positions: PyObject,
        quaternions: PyObject,
        dynamics: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        let trajectory = Trajectory::new(py, timestamps, positions, quaternions)?;
        let n = trajectory.poses_ref().len();
        let dynamics = extract_dynamics(&dynamics, n, "timestamps")?;
        Ok(Self {
            trajectory,
            dynamics,
        })
    }

    /// Construct from an existing Trajectory + (N, 12) dynamics array.
    #[staticmethod]
    fn from_trajectory_and_dynamics(
        trajectory: &Trajectory,
        dynamics: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        let n = trajectory.poses_ref().len();
        let dynamics = extract_dynamics(&dynamics, n, "trajectory")?;
        Ok(Self {
            trajectory: trajectory.clone(),
            dynamics,
        })
    }

    /// Create an empty DynamicTrajectory.
    #[staticmethod]
    fn create_empty() -> Self {
        Self {
            trajectory: Trajectory::create_empty(),
            dynamics: Vec::new(),
        }
    }

    // =========================================================================
    // Properties (delegate to inner trajectory)
    // =========================================================================

    /// Number of entries in the trajectory.
    fn __len__(&self) -> usize {
        self.trajectory.poses_ref().len()
    }

    /// Check if the trajectory is empty.
    fn is_empty(&self) -> bool {
        self.trajectory.poses_ref().is_empty()
    }

    /// Timestamps in microseconds as numpy array.
    #[getter]
    fn timestamps_us<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        let ts: Vec<u64> = self
            .trajectory
            .poses_ref()
            .iter()
            .map(|e| e.timestamp_us)
            .collect();
        PyArray1::from_vec(py, ts)
    }

    /// Get the time range as a Python range(start_us, end_us).
    #[getter]
    fn time_range_us<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let builtins = py.import("builtins")?;
        let range_cls = builtins.getattr("range")?;
        let poses = self.trajectory.poses_ref();
        if poses.is_empty() {
            range_cls.call1((0i64, 0i64))
        } else {
            let start = poses[0].timestamp_us as i64;
            let end = (poses.last().unwrap().timestamp_us + 1) as i64;
            range_cls.call1((start, end))
        }
    }

    /// Get the time range as (start_us, end_us) tuple. Returns (0, 0) if empty.
    fn get_time_range_tuple(&self) -> (u64, u64) {
        let poses = self.trajectory.poses_ref();
        if poses.is_empty() {
            (0, 0)
        } else {
            (
                poses[0].timestamp_us,
                poses.last().unwrap().timestamp_us + 1,
            )
        }
    }

    /// Positions as 2D numpy array of shape (N, 3).
    #[getter]
    fn positions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let poses = self.trajectory.poses_ref();
        let n = poses.len();
        if n == 0 {
            return PyArray2::zeros(py, [0, 3], false);
        }
        let data: Vec<Vec<f32>> = poses
            .iter()
            .map(|e| {
                let p = e.pose.position();
                vec![p.x, p.y, p.z]
            })
            .collect();
        PyArray2::from_vec2(py, &data).unwrap_or_else(|_| PyArray2::zeros(py, [n, 3], false))
    }

    /// Quaternions as 2D numpy array of shape (N, 4) in scipy format.
    #[getter]
    fn quaternions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let poses = self.trajectory.poses_ref();
        let n = poses.len();
        if n == 0 {
            return PyArray2::zeros(py, [0, 4], false);
        }
        let data: Vec<Vec<f32>> = poses
            .iter()
            .map(|e| quat_to_scipy(e.pose.quaternion()).to_vec())
            .collect();
        PyArray2::from_vec2(py, &data).unwrap_or_else(|_| PyArray2::zeros(py, [n, 4], false))
    }

    /// Get the last pose. Raises IndexError if empty.
    #[getter]
    fn last_pose(&self) -> PyResult<Pose> {
        self.trajectory
            .poses_ref()
            .last()
            .map(|e| e.pose)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    "Cannot get last_pose of empty DynamicTrajectory",
                )
            })
    }

    /// Get the first pose. Raises IndexError if empty.
    #[getter]
    fn first_pose(&self) -> PyResult<Pose> {
        self.trajectory
            .poses_ref()
            .first()
            .map(|e| e.pose)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    "Cannot get first_pose of empty DynamicTrajectory",
                )
            })
    }

    /// Get a single Pose at the given index.
    fn get_pose(&self, idx: isize) -> PyResult<Pose> {
        let poses = self.trajectory.poses_ref();
        let n = poses.len() as isize;
        let idx = if idx < 0 { n + idx } else { idx };
        if idx < 0 || idx >= n {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "index {} out of range for DynamicTrajectory of length {}",
                idx, n
            )));
        }
        Ok(poses[idx as usize].pose)
    }

    // =========================================================================
    // Dynamics access
    // =========================================================================

    /// Dynamics as 2D numpy array of shape (N, 12).
    #[getter]
    fn dynamics<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.dynamics.len();
        if n == 0 {
            return PyArray2::zeros(py, [0, 12], false);
        }
        let mut data = Vec::with_capacity(n * 12);
        for row in &self.dynamics {
            data.extend_from_slice(row);
        }
        PyArray1::from_vec(py, data)
            .reshape([n, 12])
            .expect("reshape to (N, 12) cannot fail for n*12 elements")
    }

    /// Linear interpolation of dynamics at query timestamps.
    ///
    /// Clamps outside range (same semantics as DynamicStateHistory).
    /// Returns (M, 12) f64 array.
    fn interpolate_dynamics<'py>(
        &self,
        py: Python<'py>,
        target_timestamps: PyReadonlyArray1<'_, u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let targets = target_timestamps.as_slice()?;
        let m = targets.len();
        let poses = self.trajectory.poses_ref();
        let n = poses.len();

        if n == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot interpolate dynamics on empty DynamicTrajectory",
            ));
        }

        let mut out = vec![0.0f64; m * 12];

        if n == 1 {
            for i in 0..m {
                for j in 0..12 {
                    out[i * 12 + j] = self.dynamics[0][j];
                }
            }
        } else {
            for i in 0..m {
                let t = targets[i];
                let first_ts = poses[0].timestamp_us;
                let last_ts = poses[n - 1].timestamp_us;

                if t <= first_ts {
                    for j in 0..12 {
                        out[i * 12 + j] = self.dynamics[0][j];
                    }
                } else if t >= last_ts {
                    for j in 0..12 {
                        out[i * 12 + j] = self.dynamics[n - 1][j];
                    }
                } else {
                    // Binary search for segment
                    let idx = poses
                        .partition_point(|e| e.timestamp_us <= t)
                        .saturating_sub(1)
                        .min(n - 2);

                    let t0 = poses[idx].timestamp_us;
                    let t1 = poses[idx + 1].timestamp_us;
                    let alpha = if t1 > t0 {
                        (t - t0) as f64 / (t1 - t0) as f64
                    } else {
                        0.0
                    };

                    for j in 0..12 {
                        let v0 = self.dynamics[idx][j];
                        let v1 = self.dynamics[idx + 1][j];
                        out[i * 12 + j] = v0 + alpha * (v1 - v0);
                    }
                }
            }
        }

        Ok(PyArray1::from_vec(py, out)
            .reshape([m, 12])
            .expect("reshape to (M, 12) cannot fail"))
    }

    // =========================================================================
    // Trajectory extraction
    // =========================================================================

    /// Returns a plain Trajectory (clones the poses, drops dynamics).
    fn trajectory(&self) -> Trajectory {
        self.trajectory.clone()
    }

    // =========================================================================
    // Mutation
    // =========================================================================

    /// Append one entry. Validates timestamp > last, dynamics length == 12.
    fn update_absolute(
        &mut self,
        timestamp: u64,
        pose: &Pose,
        dynamics: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let dyn_slice = dynamics.as_slice()?;
        if dyn_slice.len() != 12 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "dynamics must have 12 elements, got {}",
                dyn_slice.len()
            )));
        }

        // Delegate timestamp validation and pose insertion to Trajectory.
        self.trajectory.update_absolute(timestamp, pose)?;

        let mut dyn_row = [0.0f64; 12];
        dyn_row.copy_from_slice(dyn_slice);
        self.dynamics.push(dyn_row);

        Ok(())
    }

    // =========================================================================
    // Combining
    // =========================================================================

    /// Concatenate: other must start after self ends.
    fn concat(&self, other: &DynamicTrajectory) -> PyResult<Self> {
        let new_traj = self.trajectory.concat(&other.trajectory)?;
        let mut new_dynamics = self.dynamics.clone();
        new_dynamics.extend_from_slice(&other.dynamics);
        Ok(Self {
            trajectory: new_traj,
            dynamics: new_dynamics,
        })
    }

    /// Append: handles overlapping single endpoint.
    fn append(&self, other: &DynamicTrajectory) -> PyResult<Self> {
        if self.trajectory.poses_ref().is_empty() {
            return Ok(other.clone());
        }
        if other.trajectory.poses_ref().is_empty() {
            return Ok(self.clone());
        }

        let overlap = self.trajectory.poses_ref().last().unwrap().timestamp_us
            == other.trajectory.poses_ref().first().unwrap().timestamp_us;

        let new_traj = self.trajectory.append(&other.trajectory)?;

        let mut new_dynamics = self.dynamics.clone();
        if overlap {
            // Trajectory.append skips the first pose of other on overlap.
            new_dynamics.extend_from_slice(&other.dynamics[1..]);
        } else {
            new_dynamics.extend_from_slice(&other.dynamics);
        }

        Ok(Self {
            trajectory: new_traj,
            dynamics: new_dynamics,
        })
    }

    // =========================================================================
    // Transform
    // =========================================================================

    /// Transform poses, leaves dynamics unchanged.
    #[pyo3(signature = (transform, is_relative = false))]
    fn transform(&self, transform: &Pose, is_relative: bool) -> Self {
        Self {
            trajectory: self.trajectory.transform(transform, is_relative),
            dynamics: self.dynamics.clone(),
        }
    }

    // =========================================================================
    // Pose interpolation (delegates to inner Trajectory)
    // =========================================================================

    /// Interpolate a single pose at the given timestamp.
    ///
    /// Same semantics as ``Trajectory.interpolate_pose``: the timestamp must be
    /// within the trajectory's ``[start, end)`` range.
    fn interpolate_pose(&self, at_us: u64) -> PyResult<Pose> {
        self.trajectory.interpolate_pose_internal(at_us)
    }

    /// Compute the relative transform between two timestamps.
    ///
    /// Returns ``start_pose.inverse() @ end_pose``, identical to
    /// ``Trajectory.interpolate_delta``.
    fn interpolate_delta(&self, start_us: u64, end_us: u64) -> PyResult<Pose> {
        let start_pose = self.trajectory.interpolate_pose_internal(start_us)?;
        let end_pose = self.trajectory.interpolate_pose_internal(end_us)?;
        Ok(start_pose.inv().compose(&end_pose))
    }

    /// Interpolate poses at multiple timestamps, returning a plain Trajectory.
    ///
    /// Same semantics as ``Trajectory.interpolate``: all timestamps must lie
    /// within ``[start, end)`` of this trajectory. The dynamics are **not**
    /// interpolated — use ``interpolate_dynamics`` separately if needed.
    fn interpolate(&self, target_timestamps: PyReadonlyArray1<'_, u64>) -> PyResult<Trajectory> {
        self.trajectory.interpolate(target_timestamps)
    }

    /// Clip the trajectory to a time range, returning a plain Trajectory.
    ///
    /// Same semantics as ``Trajectory.clip``: interpolates at boundaries and
    /// includes interior poses. Dynamics are **not** preserved — use
    /// ``trajectory().clip(...)`` and reconstruct if dynamics are needed.
    fn clip(&self, start_us: u64, end_us: u64) -> PyResult<Trajectory> {
        self.trajectory.clip(start_us, end_us)
    }

    // =========================================================================
    // Debug
    // =========================================================================

    fn __repr__(&self) -> String {
        let (start, end) = self.get_time_range_tuple();
        format!(
            "DynamicTrajectory(n_poses={}, time_range_us={}..{})",
            self.trajectory.poses_ref().len(),
            start,
            end
        )
    }

    /// Create a deep copy of this DynamicTrajectory.
    #[pyo3(name = "clone")]
    fn py_clone(&self) -> Self {
        self.clone()
    }
}
