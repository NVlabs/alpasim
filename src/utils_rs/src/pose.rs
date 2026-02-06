// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 NVIDIA Corporation

//! Pose: A single rigid transform (position + quaternion).
//!
//! This is the fundamental pose type - always represents exactly one transform.
//! For batched poses with timestamps, use Trajectory.

use glam::{Mat4, Quat, Vec3};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

use crate::array_utils::{
    extract_quat_proto, extract_quat_scipy, extract_quat_scipy_denormalized, extract_vec3,
};

// Re-export quaternion conversion utilities for use by other modules
pub use crate::array_utils::{quat_to_proto, quat_to_scipy};

/// A single rigid transform: position + quaternion rotation.
///
/// This is the atomic unit for pose representation.
/// Pose is always a single transform - no batch dimension, no scalar tracking.
///
/// Quaternions are stored internally in scipy format (x, y, z, w) for
/// compatibility with scipy.spatial.transform.Rotation.
#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct Pose {
    position: Vec3,
    quaternion: Quat,
}

impl Pose {
    /// Create a Pose from glam types (internal use).
    #[inline]
    pub fn from_glam(position: Vec3, quaternion: Quat) -> Self {
        Self {
            position,
            quaternion,
        }
    }

    /// Get position as glam Vec3.
    #[inline]
    pub fn position(&self) -> Vec3 {
        self.position
    }

    /// Get quaternion as glam Quat.
    #[inline]
    pub fn quaternion(&self) -> Quat {
        self.quaternion
    }

    /// Compute pose composition: self @ other.
    #[inline]
    pub fn compose(&self, other: &Pose) -> Pose {
        // new_pos = self.quat * other.pos + self.pos
        // new_quat = self.quat * other.quat
        Pose {
            position: self.position + self.quaternion * other.position,
            quaternion: self.quaternion * other.quaternion,
        }
    }

    /// Compute the inverse of this pose.
    #[inline]
    pub fn inv(&self) -> Pose {
        let inv_quat = self.quaternion.inverse();
        Pose {
            position: -(inv_quat * self.position),
            quaternion: inv_quat,
        }
    }

    /// Get rotation as 3x3 matrix (row-major).
    #[inline]
    pub fn rotation_matrix(&self) -> [[f32; 3]; 3] {
        let mat = glam::Mat3::from_quat(self.quaternion);
        // Mat3 is column-major in glam, convert to row-major
        [
            [mat.col(0).x, mat.col(1).x, mat.col(2).x],
            [mat.col(0).y, mat.col(1).y, mat.col(2).y],
            [mat.col(0).z, mat.col(1).z, mat.col(2).z],
        ]
    }
}

#[pymethods]
impl Pose {
    /// Create a new Pose from position and quaternion numpy arrays.
    ///
    /// Args:
    ///     position: numpy array of shape (3,) with [x, y, z] position (float32 or float64)
    ///     quaternion: numpy array of shape (4,) with [x, y, z, w] quaternion in scipy format (float32 or float64)
    #[new]
    fn new(py: Python<'_>, position: PyObject, quaternion: PyObject) -> PyResult<Self> {
        let pos = extract_vec3(py, &position, "position")?;
        let quat = extract_quat_scipy(py, &quaternion, "quaternion")?;
        Ok(Self {
            position: pos,
            quaternion: quat,
        })
    }

    /// Create a pose from a possibly non-unit quaternion by normalizing it.
    ///
    /// Use this when the quaternion may not be unit-length (e.g. from an optimizer).
    /// Zero quaternions are rejected with ValueError; any other quaternion is
    /// normalized and used.
    ///
    /// Args:
    ///     position: numpy array of shape (3,) with [x, y, z] position (float32 or float64)
    ///     quaternion: numpy array of shape (4,) with [x, y, z, w] quaternion in scipy format
    #[staticmethod]
    fn from_denormalized_quat(
        py: Python<'_>,
        position: PyObject,
        quaternion: PyObject,
    ) -> PyResult<Self> {
        let pos = extract_vec3(py, &position, "position")?;
        let quat = extract_quat_scipy_denormalized(py, &quaternion, "quaternion")?;
        Ok(Self {
            position: pos,
            quaternion: quat,
        })
    }

    /// Create an identity pose (zero position, identity rotation).
    #[staticmethod]
    fn identity() -> Self {
        Self {
            position: Vec3::ZERO,
            quaternion: Quat::IDENTITY,
        }
    }

    /// Position as numpy array of shape (3,).
    #[getter]
    fn vec3<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &[self.position.x, self.position.y, self.position.z])
    }

    /// Quaternion as numpy array of shape (4,) in scipy format [x, y, z, w].
    #[getter]
    fn quat<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let q = quat_to_scipy(self.quaternion);
        PyArray1::from_slice(py, &q)
    }

    /// Compose two poses: self @ other.
    ///
    /// Result: new_position = self.rotation * other.position + self.position
    ///         new_rotation = self.rotation * other.rotation
    fn __matmul__(&self, other: &Pose) -> Pose {
        self.compose(other)
    }

    /// Compute the inverse of this pose.
    ///
    /// If self transforms A -> B, then inverse transforms B -> A.
    fn inverse(&self) -> Pose {
        self.inv()
    }

    /// Extract yaw angle (rotation around Z axis) in radians.
    fn yaw(&self) -> f32 {
        let q = self.quaternion;
        // yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        (2.0 * (q.w * q.z + q.x * q.y)).atan2(1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    }

    /// Convert to proto format: ([x, y, z], [w, x, y, z]).
    ///
    /// Returns position and quaternion in gRPC proto order.
    /// Note: Proto quaternion order is (w, x, y, z), different from scipy.
    fn to_proto(&self) -> ([f32; 3], [f32; 4]) {
        (
            [self.position.x, self.position.y, self.position.z],
            quat_to_proto(self.quaternion),
        )
    }

    /// Create from proto format.
    ///
    /// Args:
    ///     position: numpy array of shape (3,) with [x, y, z] position
    ///     quat_wxyz: numpy array of shape (4,) with [w, x, y, z] quaternion in proto order
    #[staticmethod]
    fn from_proto(py: Python<'_>, position: PyObject, quat_wxyz: PyObject) -> PyResult<Self> {
        let pos = extract_vec3(py, &position, "position")?;
        let quat = extract_quat_proto(py, &quat_wxyz, "quat_wxyz")?;
        Ok(Self {
            position: pos,
            quaternion: quat,
        })
    }

    /// Convert to SE3 4x4 matrix as numpy array.
    fn as_se3<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let mat = Mat4::from_rotation_translation(self.quaternion, self.position);
        let cols = mat.to_cols_array_2d();
        // Mat4 is column-major, but we want row-major for numpy compatibility
        // with the existing Python implementation
        let mut rows = [[0.0f32; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                rows[i][j] = cols[j][i];
            }
        }
        PyArray2::from_vec2(py, &rows.map(|r| r.to_vec()).to_vec())
            .expect("Failed to create SE3 array")
    }

    /// Create from SE3 4x4 matrix.
    #[staticmethod]
    fn from_se3(mat: PyReadonlyArray2<'_, f32>) -> PyResult<Self> {
        let shape = mat.shape();
        if shape != [4, 4] {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "SE3 matrix must be 4x4, got {:?}",
                shape
            )));
        }

        let slice = mat.as_slice()?;
        // Input is row-major, convert to column-major for Mat4
        let mut cols = [[0.0f32; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                cols[j][i] = slice[i * 4 + j];
            }
        }
        let mat4 = Mat4::from_cols_array_2d(&cols);

        // Extract rotation and translation
        let (_, rotation, translation) = mat4.to_scale_rotation_translation();

        Ok(Self {
            position: translation,
            quaternion: rotation,
        })
    }

    /// String representation for debugging.
    fn __repr__(&self) -> String {
        format!(
            "Pose(vec3=[{:.3}, {:.3}, {:.3}], yaw={:.3}rad)",
            self.position.x,
            self.position.y,
            self.position.z,
            self.yaw()
        )
    }

    /// Create a copy of this pose.
    #[pyo3(name = "clone")]
    fn py_clone(&self) -> Self {
        *self
    }

    /// Check equality with another pose (approximate, for floating point).
    fn __eq__(&self, other: &Pose) -> bool {
        const EPSILON: f32 = 1e-6;
        (self.position - other.position).length() < EPSILON
            && (self.quaternion.x - other.quaternion.x).abs() < EPSILON
            && (self.quaternion.y - other.quaternion.y).abs() < EPSILON
            && (self.quaternion.z - other.quaternion.z).abs() < EPSILON
            && (self.quaternion.w - other.quaternion.w).abs() < EPSILON
    }

}
