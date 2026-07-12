// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 NVIDIA Corporation

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};

const WIRE_FIXED64: u32 = 1;
const WIRE_LENGTH_DELIMITED: u32 = 2;
const WIRE_FIXED32: u32 = 5;

#[pyfunction]
pub fn build_drive_response_bytes<'py>(
    py: Python<'py>,
    time_now_us: i64,
    xyz: PyReadonlyArray2<'_, f32>,
    quat_wxyz: PyReadonlyArray2<'_, f32>,
    dt_us: PyReadonlyArray1<'_, i64>,
) -> PyResult<Bound<'py, PyBytes>> {
    let xyz_shape = xyz.shape();
    let quat_shape = quat_wxyz.shape();
    let dt_shape = dt_us.shape();
    if xyz_shape.len() != 2 || xyz_shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "xyz must have shape (N, 3), got {:?}",
            xyz_shape
        )));
    }
    if quat_shape.len() != 2 || quat_shape[1] != 4 {
        return Err(PyValueError::new_err(format!(
            "quat_wxyz must have shape (N, 4), got {:?}",
            quat_shape
        )));
    }
    if dt_shape.len() != 1 {
        return Err(PyValueError::new_err(format!(
            "dt_us must have shape (N,), got {:?}",
            dt_shape
        )));
    }

    let horizon = xyz_shape[0];
    if quat_shape[0] != horizon || dt_shape[0] != horizon {
        return Err(PyValueError::new_err(format!(
            "mismatched leading dims: xyz={:?}, quat_wxyz={:?}, dt_us={:?}",
            xyz_shape, quat_shape, dt_shape
        )));
    }

    let xyz = xyz.as_slice()?;
    let quat_wxyz = quat_wxyz.as_slice()?;
    let dt_us = dt_us.as_slice()?;

    let mut trajectory = Vec::with_capacity(horizon * 96);
    for i in 0..horizon {
        let timestamp_us = checked_timestamp_us(time_now_us, dt_us[i])?;
        let pose = encode_pose(&xyz[i * 3..i * 3 + 3], &quat_wxyz[i * 4..i * 4 + 4]);
        let mut pose_at_time = Vec::with_capacity(pose.len() + 16);
        put_message_field(&mut pose_at_time, 1, &pose);
        put_fixed64_field(&mut pose_at_time, 2, timestamp_us);
        put_message_field(&mut trajectory, 1, &pose_at_time);
    }

    let mut response = Vec::with_capacity(trajectory.len() + 8);
    if !trajectory.is_empty() {
        put_message_field(&mut response, 2, &trajectory);
    }
    Ok(PyBytes::new(py, &response))
}

#[pyfunction]
pub fn pack_trajectory_arrays<'py>(
    py: Python<'py>,
    trajectory: &Bound<'py, PyAny>,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
)> {
    let poses = trajectory.getattr("poses")?;
    let pose_count = poses.len()?;
    let mut rows = Vec::with_capacity(pose_count);
    for i in 0..pose_count {
        let pose_at_time = poses.get_item(i)?;
        let timestamp_us = pose_at_time.getattr("timestamp_us")?.extract::<i64>()?;
        let pose = pose_at_time.getattr("pose")?;
        let vec = pose.getattr("vec")?;
        let quat = pose.getattr("quat")?;
        rows.push(PoseRow {
            timestamp_us,
            xyz: [
                vec.getattr("x")?.extract::<f32>()?,
                vec.getattr("y")?.extract::<f32>()?,
                vec.getattr("z")?.extract::<f32>()?,
            ],
            quat_wxyz: [
                quat.getattr("w")?.extract::<f32>()?,
                quat.getattr("x")?.extract::<f32>()?,
                quat.getattr("y")?.extract::<f32>()?,
                quat.getattr("z")?.extract::<f32>()?,
            ],
        });
    }
    rows.sort_by_key(|row| row.timestamp_us);

    let mut timestamps = Vec::with_capacity(pose_count);
    let mut xyz = Vec::with_capacity(pose_count * 3);
    let mut quat_wxyz = Vec::with_capacity(pose_count * 4);
    for row in rows {
        timestamps.push(row.timestamp_us);
        xyz.extend_from_slice(&row.xyz);
        quat_wxyz.extend_from_slice(&row.quat_wxyz);
    }

    Ok((
        PyArray1::from_vec(py, timestamps),
        array2_from_shape_vec(py, pose_count, 3, xyz)?,
        array2_from_shape_vec(py, pose_count, 4, quat_wxyz)?,
    ))
}

#[pyfunction]
pub fn pack_route_array<'py>(
    py: Python<'py>,
    route: &Bound<'py, PyAny>,
) -> PyResult<(i64, Bound<'py, PyArray2<f32>>)> {
    let waypoints = route.getattr("waypoints")?;
    let waypoint_count = waypoints.len()?;
    let mut xyz = Vec::with_capacity(waypoint_count * 3);
    for i in 0..waypoint_count {
        let waypoint = waypoints.get_item(i)?;
        xyz.push(waypoint.getattr("x")?.extract::<f32>()?);
        xyz.push(waypoint.getattr("y")?.extract::<f32>()?);
        xyz.push(waypoint.getattr("z")?.extract::<f32>()?);
    }
    Ok((
        route.getattr("timestamp_us")?.extract::<i64>()?,
        array2_from_shape_vec(py, waypoint_count, 3, xyz)?,
    ))
}

struct PoseRow {
    timestamp_us: i64,
    xyz: [f32; 3],
    quat_wxyz: [f32; 4],
}

fn array2_from_shape_vec<'py>(
    py: Python<'py>,
    rows: usize,
    cols: usize,
    values: Vec<f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let array = Array2::from_shape_vec((rows, cols), values)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyArray2::from_owned_array(py, array))
}

fn checked_timestamp_us(time_now_us: i64, dt_us: i64) -> PyResult<u64> {
    let timestamp_us = time_now_us
        .checked_add(dt_us)
        .ok_or_else(|| PyValueError::new_err("timestamp_us overflow"))?;
    if timestamp_us < 0 {
        return Err(PyValueError::new_err(format!(
            "timestamp_us must be non-negative, got {}",
            timestamp_us
        )));
    }
    Ok(timestamp_us as u64)
}

fn encode_pose(xyz: &[f32], quat_wxyz: &[f32]) -> Vec<u8> {
    let mut vec3 = Vec::with_capacity(15);
    put_float_field(&mut vec3, 1, xyz[0]);
    put_float_field(&mut vec3, 2, xyz[1]);
    put_float_field(&mut vec3, 3, xyz[2]);

    let mut quat = Vec::with_capacity(20);
    put_float_field(&mut quat, 1, quat_wxyz[0]);
    put_float_field(&mut quat, 2, quat_wxyz[1]);
    put_float_field(&mut quat, 3, quat_wxyz[2]);
    put_float_field(&mut quat, 4, quat_wxyz[3]);

    let mut pose = Vec::with_capacity(vec3.len() + quat.len() + 8);
    put_message_field(&mut pose, 1, &vec3);
    put_message_field(&mut pose, 2, &quat);
    pose
}

fn put_float_field(buf: &mut Vec<u8>, field_number: u32, value: f32) {
    if value.to_bits() == 0 {
        return;
    }
    put_key(buf, field_number, WIRE_FIXED32);
    buf.extend_from_slice(&value.to_le_bytes());
}

fn put_fixed64_field(buf: &mut Vec<u8>, field_number: u32, value: u64) {
    if value == 0 {
        return;
    }
    put_key(buf, field_number, WIRE_FIXED64);
    buf.extend_from_slice(&value.to_le_bytes());
}

fn put_message_field(buf: &mut Vec<u8>, field_number: u32, value: &[u8]) {
    put_key(buf, field_number, WIRE_LENGTH_DELIMITED);
    put_varint(buf, value.len() as u64);
    buf.extend_from_slice(value);
}

fn put_key(buf: &mut Vec<u8>, field_number: u32, wire_type: u32) {
    put_varint(buf, ((field_number << 3) | wire_type) as u64);
}

fn put_varint(buf: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        buf.push((value as u8) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
}
