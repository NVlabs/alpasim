// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 NVIDIA Corporation

//! Shared utilities for extracting numpy arrays with flexible dtype handling.
//!
//! These helpers accept multiple numeric dtypes and convert to the internal
//! representation as needed, with clear error messages for invalid dtypes.

use glam::{Quat, Vec3};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

// ============================================================================
// Quaternion conversion utilities
// ============================================================================

/// Convert scipy quaternion (x, y, z, w) to glam quaternion.
#[inline]
pub fn quat_from_scipy(x: f32, y: f32, z: f32, w: f32) -> Quat {
    Quat::from_xyzw(x, y, z, w)
}

/// Convert glam quaternion to scipy format array [x, y, z, w].
#[inline]
pub fn quat_to_scipy(q: Quat) -> [f32; 4] {
    [q.x, q.y, q.z, q.w]
}

/// Convert proto quaternion (w, x, y, z) to glam quaternion.
#[inline]
pub fn quat_from_proto(w: f32, x: f32, y: f32, z: f32) -> Quat {
    Quat::from_xyzw(x, y, z, w)
}

/// Convert glam quaternion to proto format array [w, x, y, z].
#[inline]
pub fn quat_to_proto(q: Quat) -> [f32; 4] {
    [q.w, q.x, q.y, q.z]
}

/// Maximum allowed deviation of quaternion length² from 1.0.
///
/// Quaternions within this tolerance are silently normalized to correct for
/// floating-point drift (e.g. f64→f32 conversion, serialization round-trips).
/// Quaternions outside this tolerance are rejected as likely bugs in calling code.
const UNIT_QUAT_LEN_SQ_TOL: f32 = 1e-4;

/// Ensure a quaternion is unit-length, normalizing if within tolerance.
///
/// - Near-zero quaternions (length² < 1e-12) are rejected as corrupt data.
/// - Quaternions within [`UNIT_QUAT_LEN_SQ_TOL`] of unit length are silently
///   normalized and returned.
/// - Quaternions far from unit length are rejected with a clear error message.
pub fn ensure_unit_quat(q: Quat, name: &str) -> PyResult<Quat> {
    let len_sq = q.length_squared();

    // Near-zero: corrupt or uninitialized data
    if len_sq < 1e-12 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{}: quaternion is near-zero (length²={:.2e}), \
             this indicates corrupt or uninitialized data",
            name, len_sq
        )));
    }

    // Close to unit: normalize and return
    if (len_sq - 1.0).abs() <= UNIT_QUAT_LEN_SQ_TOL {
        return Ok(q.normalize());
    }

    // Far from unit: likely a caller bug
    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
        "{}: quaternion is not unit-length (length={:.6}). \
         Normalize before passing to Pose/Trajectory.",
        name,
        len_sq.sqrt()
    )))
}

/// Normalize a quaternion, rejecting zero (for use by from_denormalized_quat).
///
/// Returns Ok(q.normalize()) for any non-zero quaternion, Err for near-zero.
pub fn normalize_quat_or_reject_zero(q: Quat, name: &str) -> PyResult<Quat> {
    let len_sq = q.length_squared();
    if len_sq < 1e-12 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{}: quaternion is near-zero (length²={:.2e}), cannot normalize",
            name, len_sq
        )));
    }
    Ok(q.normalize())
}

// ============================================================================
// Generic array extraction (1D -> Vec<f32>)
// ============================================================================
//
// TODO(perf): These helpers always copy (to_vec/collect). Callers that take
// PyObject could instead try PyReadonlyArray1<f32> first and use as_slice()
// when dtype matches, copying only when converting f64->f32 or for invalid
// dtypes, to avoid allocation for the common float32 case.

/// Extract a 1D array as Vec<f32>, accepting f32 or f64.
pub fn extract_array1_f32(py: Python<'_>, obj: &PyObject, name: &str) -> PyResult<Vec<f32>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, f32>>(py) {
        return Ok(arr.as_slice()?.to_vec());
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, f64>>(py) {
        return Ok(arr.as_slice()?.iter().map(|&v| v as f32).collect());
    }
    Err(dtype_error(py, obj, name, "float32 or float64"))
}

/// Extract a 1D array as Vec<u64>, accepting unsigned integer types only.
///
/// Signed types (int32, int64) are explicitly rejected to avoid silent
/// wrapping of negative values into large u64 values.
pub fn extract_array1_u64(py: Python<'_>, obj: &PyObject, name: &str) -> PyResult<Vec<u64>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, u64>>(py) {
        return Ok(arr.as_slice()?.to_vec());
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, u32>>(py) {
        return Ok(arr.as_slice()?.iter().map(|&v| v as u64).collect());
    }
    // Reject signed types explicitly with a helpful message.
    let dtype_str = obj
        .getattr(py, "dtype")
        .and_then(|d| d.bind(py).str().map(|s| s.to_string()))
        .unwrap_or_else(|_| "unknown".to_string());
    if dtype_str.contains("int") && !dtype_str.contains("uint") {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "{} has signed dtype {} which is not allowed for u64 conversion \
             (negative values would silently wrap). Use .astype(np.uint64) after \
             validating all values are non-negative.",
            name, dtype_str
        )));
    }
    Err(dtype_error(py, obj, name, "uint64 or uint32"))
}

// ============================================================================
// Generic array extraction (2D -> Vec + shape)
// ============================================================================

/// Extract a 2D array as (flat Vec<f32>, [rows, cols]), accepting f32 or f64.
pub fn extract_array2_f32(
    py: Python<'_>,
    obj: &PyObject,
    name: &str,
) -> PyResult<(Vec<f32>, [usize; 2])> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, f32>>(py) {
        let shape = arr.shape();
        let view = arr.as_array();
        let mut data = Vec::with_capacity(shape[0] * shape[1]);
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                data.push(view[[i, j]]);
            }
        }
        return Ok((data, [shape[0], shape[1]]));
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, f64>>(py) {
        let shape = arr.shape();
        let view = arr.as_array();
        let mut data = Vec::with_capacity(shape[0] * shape[1]);
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                data.push(view[[i, j]] as f32);
            }
        }
        return Ok((data, [shape[0], shape[1]]));
    }

    // Build an error that mentions 1D when appropriate
    let ndim = obj
        .getattr(py, "ndim")
        .and_then(|d| d.extract::<usize>(py))
        .unwrap_or(0);
    let dtype_str = obj
        .getattr(py, "dtype")
        .and_then(|d| d.bind(py).str().map(|s| s.to_string()))
        .unwrap_or_else(|_| "unknown".to_string());

    let hint = if ndim == 1 {
        "For 1D arrays, use arr[np.newaxis, :] or arr.reshape(1, -1)"
    } else if dtype_str.contains("int") {
        "Use .astype(float) or np.array(..., dtype=float) to convert"
    } else {
        "Expected 2D float array"
    };

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "{} must be a 2D numpy array with dtype float32 or float64, got ndim={} dtype={}. Hint: {}",
        name, ndim, dtype_str, hint
    )))
}

// ============================================================================
// Higher-level extractors (for common patterns)
// ============================================================================

/// Extract a Vec3 from a 1D array of length 3.
pub fn extract_vec3(py: Python<'_>, obj: &PyObject, name: &str) -> PyResult<Vec3> {
    let data = extract_array1_f32(py, obj, name)?;
    if data.len() != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{} must have 3 elements, got {}",
            name,
            data.len()
        )));
    }
    Ok(Vec3::new(data[0], data[1], data[2]))
}

/// Extract a quaternion in scipy format (x, y, z, w) from a 1D array of length 4.
pub fn extract_quat_scipy(py: Python<'_>, obj: &PyObject, name: &str) -> PyResult<Quat> {
    let data = extract_array1_f32(py, obj, name)?;
    if data.len() != 4 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{} must have 4 elements, got {}",
            name,
            data.len()
        )));
    }
    let q = quat_from_scipy(data[0], data[1], data[2], data[3]);
    ensure_unit_quat(q, name)
}

/// Extract a quaternion in proto format (w, x, y, z) from a 1D array of length 4.
pub fn extract_quat_proto(py: Python<'_>, obj: &PyObject, name: &str) -> PyResult<Quat> {
    let data = extract_array1_f32(py, obj, name)?;
    if data.len() != 4 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{} must have 4 elements, got {}",
            name,
            data.len()
        )));
    }
    let q = quat_from_proto(data[0], data[1], data[2], data[3]);
    ensure_unit_quat(q, name)
}

/// Extract a quaternion in scipy format and normalize (for from_denormalized_quat).
/// Rejects zero quaternions; any other quaternion is normalized and returned.
pub fn extract_quat_scipy_denormalized(
    py: Python<'_>,
    obj: &PyObject,
    name: &str,
) -> PyResult<Quat> {
    let data = extract_array1_f32(py, obj, name)?;
    if data.len() != 4 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{} must have 4 elements, got {}",
            name,
            data.len()
        )));
    }
    let q = quat_from_scipy(data[0], data[1], data[2], data[3]);
    normalize_quat_or_reject_zero(q, name)
}

// ============================================================================
// Error helper
// ============================================================================

fn dtype_error(py: Python<'_>, obj: &PyObject, name: &str, expected: &str) -> PyErr {
    let dtype_str = obj
        .getattr(py, "dtype")
        .and_then(|d| d.bind(py).str().map(|s| s.to_string()))
        .unwrap_or_else(|_| "unknown".to_string());
    PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "{} must be a numpy array with dtype {}, got dtype={}",
        name, expected, dtype_str
    ))
}
