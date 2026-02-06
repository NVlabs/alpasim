// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 NVIDIA Corporation

//! Polyline: Rust-backed polyline with efficient point projection.
//!
//! Provides high-performance segment projection operations without
//! Python/numpy loop overhead.

use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

use crate::array_utils::{extract_array1_f32, extract_array2_f32};
use crate::pose::Pose;

/// Result of projecting a point onto a polyline segment.
/// Returns (projected_point, segment_idx, distance_along_segment).
#[derive(Clone, Debug)]
pub struct ProjectionResult {
    pub point: Vec<f32>,
    pub segment_idx: usize,
    pub distance_along: f32,
}

/// A spatial polyline as an ordered set of 2D or 3D waypoints.
///
/// Points are stored as a flat f32 array for memory efficiency.
/// Provides batch operations (projection, interpolation) entirely in Rust.
#[pyclass(name = "Polyline")]
#[derive(Clone)]
pub struct Polyline {
    /// Points as flat array [x0, y0, (z0), x1, y1, (z1), ...] (N*D elements)
    points: Vec<f32>,
    /// Spatial dimension (2 or 3)
    dimension: usize,
}

impl Polyline {
    /// Create a Polyline from pre-validated flat point data (crate-internal).
    #[inline]
    pub(crate) fn from_flat(points: Vec<f32>, dimension: usize) -> Self {
        Self { points, dimension }
    }

    /// Number of waypoints.
    #[inline]
    pub fn len(&self) -> usize {
        if self.dimension == 0 {
            0
        } else {
            self.points.len() / self.dimension
        }
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Get point at index as slice.
    #[inline]
    pub fn get_point(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        &self.points[start..start + self.dimension]
    }

    /// Compute squared distance between two points.
    #[inline]
    fn distance_squared(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum()
    }

    /// Compute distance between two points.
    #[inline]
    fn distance(a: &[f32], b: &[f32]) -> f32 {
        Self::distance_squared(a, b).sqrt()
    }

    /// Dot product of two vectors.
    #[inline]
    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Project point onto a line segment, returning (projected_point, t, distance).
    /// t is the parameter along the segment (clamped to [0, segment_length]).
    fn project_point_to_segment_impl(
        point: &[f32],
        segment_start: &[f32],
        segment_end: &[f32],
    ) -> (Vec<f32>, f32, f32) {
        let dim = point.len();

        // Compute segment vector
        let segment_vec: Vec<f32> = segment_end
            .iter()
            .zip(segment_start.iter())
            .map(|(e, s)| e - s)
            .collect();

        let segment_length_sq = Self::dot(&segment_vec, &segment_vec);
        let segment_length = segment_length_sq.sqrt();

        // Degenerate segment case
        if segment_length < 1e-6 {
            let dist = Self::distance(point, segment_start);
            return (segment_start.to_vec(), 0.0, dist);
        }

        // Compute segment direction
        let inv_len = 1.0 / segment_length;
        let segment_dir: Vec<f32> = segment_vec.iter().map(|v| v * inv_len).collect();

        // Vector from segment start to point
        let to_point: Vec<f32> = point
            .iter()
            .zip(segment_start.iter())
            .map(|(p, s)| p - s)
            .collect();

        // Project onto segment direction
        let t = Self::dot(&to_point, &segment_dir);
        let t_clamped = t.clamp(0.0, segment_length);

        // Compute projected point
        let projected: Vec<f32> = (0..dim)
            .map(|i| segment_start[i] + t_clamped * segment_dir[i])
            .collect();

        // Compute distance from point to projection
        let distance = Self::distance(&projected, point);

        (projected, t_clamped, distance)
    }

    /// Find the closest projection of a point onto the polyline.
    pub fn project_point_impl(&self, point: &[f32]) -> ProjectionResult {
        let n = self.len();

        if n == 0 {
            panic!("Cannot project onto empty polyline");
        }

        if n == 1 {
            return ProjectionResult {
                point: self.get_point(0).to_vec(),
                segment_idx: 0,
                distance_along: 0.0,
            };
        }

        let mut min_distance = f32::INFINITY;
        let mut best_projection = self.get_point(0).to_vec();
        let mut best_index = 0usize;
        let mut best_distance_along = 0.0f32;

        for i in 0..(n - 1) {
            let seg_start = self.get_point(i);
            let seg_end = self.get_point(i + 1);

            let (projected, distance_along, distance) =
                Self::project_point_to_segment_impl(point, seg_start, seg_end);

            if distance < min_distance {
                min_distance = distance;
                best_projection = projected;
                best_index = i;
                best_distance_along = distance_along;
            }
        }

        ProjectionResult {
            point: best_projection,
            segment_idx: best_index,
            distance_along: best_distance_along,
        }
    }

    /// Compute segment lengths (distances between consecutive waypoints).
    pub fn segment_lengths_impl(&self) -> Vec<f32> {
        let n = self.len();
        if n < 2 {
            return Vec::new();
        }

        (0..(n - 1))
            .map(|i| Self::distance(self.get_point(i), self.get_point(i + 1)))
            .collect()
    }

    /// Compute cumulative arc lengths along the polyline.
    pub fn arc_lengths_impl(&self) -> Vec<f32> {
        let seg_lengths = self.segment_lengths_impl();
        if seg_lengths.is_empty() {
            return vec![0.0; self.len().max(1)];
        }

        let mut arc_lengths = Vec::with_capacity(seg_lengths.len() + 1);
        arc_lengths.push(0.0);

        let mut cumsum = 0.0;
        for len in seg_lengths {
            cumsum += len;
            arc_lengths.push(cumsum);
        }

        arc_lengths
    }

    /// Build remaining polyline from projection point - helper for reuse.
    fn build_remaining_points(&self, point_slice: &[f32]) -> (Vec<f32>, ProjectionResult) {
        let n = self.len();

        if n == 0 {
            return (Vec::new(), ProjectionResult {
                point: vec![0.0; self.dimension],
                segment_idx: 0,
                distance_along: 0.0,
            });
        }

        let projection = self.project_point_impl(point_slice);

        if n == 1 {
            return (self.get_point(0).to_vec(), projection);
        }

        let mut remaining_points = Vec::new();

        if projection.segment_idx == n - 2 {
            let last_waypoint = self.get_point(n - 1);
            let second_last = self.get_point(n - 2);
            let segment_length = Self::distance(second_last, last_waypoint);

            if segment_length > 0.0 {
                let to_point: Vec<f32> = point_slice
                    .iter()
                    .zip(second_last.iter())
                    .map(|(p, s)| p - s)
                    .collect();
                let segment_vec: Vec<f32> = last_waypoint
                    .iter()
                    .zip(second_last.iter())
                    .map(|(e, s)| e - s)
                    .collect();
                let segment_dir: Vec<f32> = segment_vec.iter().map(|v| v / segment_length).collect();
                let t_unclamped = Self::dot(&to_point, &segment_dir);

                if t_unclamped > segment_length + 1e-6 {
                    return (Vec::new(), projection);
                }
            }

            let is_on_last = Self::distance(&projection.point, last_waypoint) < 1e-10;
            if is_on_last {
                remaining_points.extend(last_waypoint);
            } else {
                remaining_points.extend(&projection.point);
                remaining_points.extend(last_waypoint);
            }
        } else {
            remaining_points.extend(&projection.point);
            for i in (projection.segment_idx + 1)..n {
                remaining_points.extend(self.get_point(i));
            }
        }

        (remaining_points, projection)
    }
}

#[pymethods]
impl Polyline {
    /// Create a new Polyline from a numpy array of shape (N, D).
    ///
    /// Args:
    ///     points: 2D array of shape (N, D) where D is 2 or 3 (float32 or float64).
    #[new]
    fn new(py: Python<'_>, points: PyObject) -> PyResult<Self> {
        let (data, shape) = extract_array2_f32(py, &points, "points")?;
        let dimension = shape[1];
        if dimension != 2 && dimension != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "points must have 2 or 3 columns, got {}",
                dimension
            )));
        }
        Ok(Self { points: data, dimension })
    }

    /// Create an empty Polyline with the specified dimension (default 3).
    #[staticmethod]
    #[pyo3(signature = (dimension=3))]
    fn create_empty(dimension: usize) -> PyResult<Self> {
        if dimension != 2 && dimension != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "dimension must be 2 or 3, got {}",
                dimension
            )));
        }
        Ok(Self {
            points: Vec::new(),
            dimension,
        })
    }

    /// Number of waypoints.
    fn __len__(&self) -> usize {
        self.len()
    }

    /// String representation.
    fn __repr__(&self) -> String {
        let length = {
            let arc_lengths = self.arc_lengths_impl();
            if arc_lengths.is_empty() {
                0.0
            } else {
                arc_lengths[arc_lengths.len() - 1]
            }
        };
        format!(
            "Polyline(n_points={}, dimension={}, length={:.2}m)",
            self.len(),
            self.dimension,
            length
        )
    }

    // =========================================================================
    // Properties (clean API without _py suffix)
    // =========================================================================

    /// Whether the polyline contains zero waypoints.
    #[getter]
    #[pyo3(name = "is_empty")]
    fn is_empty_prop(&self) -> bool {
        self.is_empty()
    }

    /// Spatial dimensionality of the polyline (2 or 3).
    #[getter]
    #[pyo3(name = "dimension")]
    fn dimension_getter(&self) -> usize {
        self.dimension
    }

    /// Reference to the underlying waypoint array (alias for points).
    #[getter]
    fn waypoints<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.points_getter(py)
    }

    /// Get points as numpy array of shape (N, D).
    #[getter]
    #[pyo3(name = "points")]
    fn points_getter<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let n = self.len();
        if n == 0 {
            return Ok(PyArray2::zeros(py, [0, self.dimension], false));
        }

        PyArray2::from_vec2(
            py,
            &self
                .points
                .chunks(self.dimension)
                .map(|c| c.to_vec())
                .collect::<Vec<_>>(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!(
            "Failed to create numpy array from polyline points: {e}"
        )))
    }

    /// Total arc length of the polyline.
    #[getter]
    fn total_length(&self) -> f32 {
        let arc_lengths = self.arc_lengths_impl();
        if arc_lengths.is_empty() {
            0.0
        } else {
            arc_lengths[arc_lengths.len() - 1]
        }
    }

    /// Euclidean distance between consecutive waypoints.
    #[getter]
    fn segment_lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.segment_lengths_impl())
    }

    // =========================================================================
    // Methods (clean API)
    // =========================================================================

    /// Cumulative arc lengths along the polyline.
    fn arc_lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.arc_lengths_impl())
    }

    /// Orthogonally project a point onto the polyline segments.
    ///
    /// Returns (projected_point, segment_idx, distance_along_segment).
    fn project_point<'py>(
        &self,
        py: Python<'py>,
        point: PyObject,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, usize, f32)> {
        let point_slice = extract_array1_f32(py, &point, "point")?;

        if point_slice.len() != self.dimension {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "point dimension {} does not match polyline dimension {}",
                point_slice.len(),
                self.dimension
            )));
        }

        if self.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot project onto empty polyline",
            ));
        }

        let result = self.project_point_impl(&point_slice);
        let projected = PyArray1::from_vec(py, result.point);

        Ok((projected, result.segment_idx, result.distance_along))
    }

    /// Project multiple points onto the polyline in batch.
    ///
    /// Args:
    ///     points: 2D array of shape (M, D) with points to project.
    ///
    /// Returns:
    ///     Tuple of (projected_points, segment_indices, distances_along).
    fn project_points_batch<'py>(
        &self,
        py: Python<'py>,
        points: PyObject,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<usize>>,
        Bound<'py, PyArray1<f32>>,
    )> {
        let (data, shape) = extract_array2_f32(py, &points, "points")?;
        if shape[1] != self.dimension {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "points dimension {} does not match polyline dimension {}",
                shape[1], self.dimension
            )));
        }

        if self.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot project onto empty polyline",
            ));
        }

        let n_points = shape[0];
        let points_slice = data.as_slice();

        let mut projected_points = Vec::with_capacity(n_points * self.dimension);
        let mut segment_indices = Vec::with_capacity(n_points);
        let mut distances_along = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let point = &points_slice[i * self.dimension..(i + 1) * self.dimension];
            let result = self.project_point_impl(point);

            projected_points.extend(result.point);
            segment_indices.push(result.segment_idx);
            distances_along.push(result.distance_along);
        }

        let projected = PyArray2::from_vec2(
            py,
            &projected_points
                .chunks(self.dimension)
                .map(|c| c.to_vec())
                .collect::<Vec<_>>(),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        Ok((
            projected,
            PyArray1::from_vec(py, segment_indices),
            PyArray1::from_vec(py, distances_along),
        ))
    }

    /// Interpolate positions at specific distances along the polyline.
    ///
    /// Args:
    ///     distances: 1D array of distances along the polyline.
    ///
    /// Returns:
    ///     2D array of shape (M, D) with interpolated positions.
    fn positions_at<'py>(
        &self,
        py: Python<'py>,
        distances: PyObject,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let distances_slice = extract_array1_f32(py, &distances, "distances")?;

        if self.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot interpolate along an empty polyline",
            ));
        }

        let arc_lengths = self.arc_lengths_impl();

        // Handle duplicate arc lengths (zero-length segments)
        let mut unique_lengths = Vec::new();
        let mut unique_indices = Vec::new();
        for (i, &len) in arc_lengths.iter().enumerate() {
            if unique_lengths.is_empty() || len != *unique_lengths.last().unwrap() {
                unique_lengths.push(len);
                unique_indices.push(i);
            }
        }

        if unique_lengths.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot interpolate along an empty polyline",
            ));
        }

        // Validate distances are within range
        if !distances_slice.is_empty() {
            let min_dist = distances_slice.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_dist = distances_slice
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);

            if min_dist < unique_lengths[0] || max_dist > unique_lengths[unique_lengths.len() - 1] {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Requested distances must lie within the polyline arc length range",
                ));
            }
        }

        // Interpolate each dimension
        let mut result = Vec::with_capacity(distances_slice.len() * self.dimension);

        for &dist in &distances_slice {
            // Binary search for segment
            let seg_idx = match unique_lengths.binary_search_by(|&len| {
                len.partial_cmp(&dist).unwrap_or(std::cmp::Ordering::Equal)
            }) {
                Ok(i) => i.min(unique_lengths.len() - 1),
                Err(i) => (i.saturating_sub(1)).min(unique_lengths.len() - 2),
            };

            let seg_idx = seg_idx.min(unique_lengths.len() - 2);
            let next_idx = seg_idx + 1;

            let len0 = unique_lengths[seg_idx];
            let len1 = unique_lengths[next_idx];
            let idx0 = unique_indices[seg_idx];
            let idx1 = unique_indices[next_idx];

            let alpha = if (len1 - len0).abs() < 1e-10 {
                0.0
            } else {
                (dist - len0) / (len1 - len0)
            };

            let p0 = self.get_point(idx0);
            let p1 = self.get_point(idx1);

            for d in 0..self.dimension {
                result.push(p0[d] + alpha * (p1[d] - p0[d]));
            }
        }

        PyArray2::from_vec2(
            py,
            &result
                .chunks(self.dimension)
                .map(|c| c.to_vec())
                .collect::<Vec<_>>(),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))
    }

    /// Return the polyline remainder after projecting a point.
    ///
    /// Returns:
    ///     Tuple of (remaining_polyline, projection_result) where projection_result
    ///     is (projected_point, segment_idx, distance_along).
    fn remaining_from_point<'py>(
        &self,
        py: Python<'py>,
        point: PyObject,
    ) -> PyResult<(Self, (Bound<'py, PyArray1<f32>>, usize, f32))> {
        let point_slice = extract_array1_f32(py, &point, "point")?;

        if point_slice.len() != self.dimension {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "point dimension {} does not match polyline dimension {}",
                point_slice.len(),
                self.dimension
            )));
        }

        let (remaining_points, projection) = self.build_remaining_points(&point_slice);

        let remaining = Polyline {
            points: remaining_points,
            dimension: self.dimension,
        };

        let proj_point = PyArray1::from_vec(py, projection.point);
        Ok((remaining, (proj_point, projection.segment_idx, projection.distance_along)))
    }

    /// Uniformly resample the remainder of the polyline after a start point.
    ///
    /// Args:
    ///     start_point: Point to project onto the polyline.
    ///     spacing: Distance between samples.
    ///     n_points: Maximum number of points to sample.
    ///
    /// Returns:
    ///     A new Polyline with resampled points.
    fn resample_from_point(
        &self,
        py: Python<'_>,
        start_point: PyObject,
        spacing: f64,
        n_points: usize,
    ) -> PyResult<Self> {
        let spacing = spacing as f32;
        let start_slice = extract_array1_f32(py, &start_point, "start_point")?;

        if start_slice.len() != self.dimension {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "point dimension {} does not match polyline dimension {}",
                start_slice.len(),
                self.dimension
            )));
        }

        if self.is_empty() {
            return Ok(Polyline::create_empty(self.dimension)?);
        }

        let (remaining_points, _projection) = self.build_remaining_points(&start_slice);

        if remaining_points.is_empty() {
            return Ok(Polyline::create_empty(self.dimension)?);
        }

        let remaining = Polyline {
            points: remaining_points,
            dimension: self.dimension,
        };

        let arc_lengths = remaining.arc_lengths_impl();
        let total_length = arc_lengths.last().copied().unwrap_or(0.0);

        // Generate sample distances
        let sample_distances: Vec<f32> = (0..n_points)
            .map(|i| i as f32 * spacing)
            .filter(|&d| d <= total_length)
            .collect();

        if sample_distances.is_empty() {
            return Ok(Polyline::create_empty(self.dimension)?);
        }

        // Handle single-point remaining polyline: can only sample at distance 0
        if remaining.len() == 1 {
            if sample_distances[0] == 0.0 {
                return Ok(Polyline {
                    points: remaining.get_point(0).to_vec(),
                    dimension: self.dimension,
                });
            } else {
                return Ok(Polyline::create_empty(self.dimension)?);
            }
        }

        // Interpolate at sample distances
        let mut result = Vec::with_capacity(sample_distances.len() * self.dimension);

        for &dist in &sample_distances {
            // Binary search for segment
            let seg_idx = match arc_lengths.binary_search_by(|&len| {
                len.partial_cmp(&dist).unwrap_or(std::cmp::Ordering::Equal)
            }) {
                Ok(i) => i.min(arc_lengths.len() - 1),
                Err(i) => (i.saturating_sub(1)).min(arc_lengths.len().saturating_sub(2)),
            };

            let seg_idx = seg_idx.min(remaining.len().saturating_sub(2));
            let next_idx = seg_idx + 1;

            let len0 = arc_lengths[seg_idx];
            let len1 = arc_lengths[next_idx];

            let alpha = if (len1 - len0).abs() < 1e-10 {
                0.0
            } else {
                (dist - len0) / (len1 - len0)
            };

            let p0 = remaining.get_point(seg_idx);
            let p1 = remaining.get_point(next_idx);

            for d in 0..self.dimension {
                result.push(p0[d] + alpha * (p1[d] - p0[d]));
            }
        }

        Ok(Polyline {
            points: result,
            dimension: self.dimension,
        })
    }

    /// Return a copy over the waypoint slice [start:end].
    #[pyo3(signature = (start=None, end=None))]
    fn clip(&self, start: Option<usize>, end: Option<usize>) -> PyResult<Self> {
        let n = self.len();
        let start = start.unwrap_or(0);
        let end = end.unwrap_or(n);

        let start = start.min(n);
        let end = end.min(n);

        if start >= end {
            return Polyline::create_empty(self.dimension);
        }

        Ok(Self {
            points: self.points[start * self.dimension..end * self.dimension].to_vec(),
            dimension: self.dimension,
        })
    }

    /// Concatenate another polyline with matching dimensionality.
    fn append(&self, other: &Polyline) -> PyResult<Self> {
        if self.dimension != other.dimension {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot append polylines of different dimensions",
            ));
        }

        if self.is_empty() {
            return Ok(other.clone());
        }
        if other.is_empty() {
            return Ok(self.clone());
        }

        let mut points = self.points.clone();
        points.extend(&other.points);

        Ok(Self {
            points,
            dimension: self.dimension,
        })
    }

    /// Downsample ensuring minimum distance between waypoints (mutates self).
    fn downsample_with_min_distance(&mut self, min_distance: f64) {
        let min_distance = min_distance as f32;
        if self.len() < 2 {
            return;
        }

        let mut keep_points = Vec::new();
        keep_points.extend(self.get_point(0));
        let mut last_kept = self.get_point(0).to_vec();

        for i in 1..self.len() {
            let current = self.get_point(i);
            if Self::distance(&last_kept, current) >= min_distance {
                keep_points.extend(current);
                last_kept = current.to_vec();
            }
        }

        self.points = keep_points;
    }

    /// Clone the polyline.
    #[pyo3(name = "clone")]
    fn py_clone(&self) -> Self {
        self.clone()
    }

    /// Cumulative distances along the remainder of the polyline from a projected point.
    ///
    /// Returns (cumulative_distances, distance_to_projection).
    fn get_cumulative_distances_from_point<'py>(
        &self,
        py: Python<'py>,
        point: PyObject,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, f32)> {
        let point_slice = extract_array1_f32(py, &point, "point")?;

        let cumulative = self.arc_lengths_impl();
        if cumulative.is_empty() {
            return Ok((PyArray1::from_vec(py, Vec::new()), 0.0));
        }

        if point_slice.len() != self.dimension {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "point dimension {} does not match polyline dimension {}",
                point_slice.len(),
                self.dimension
            )));
        }

        let projection = self.project_point_impl(&point_slice);
        let distance_to_projection = cumulative[projection.segment_idx] + projection.distance_along;

        let (remaining_points, _) = self.build_remaining_points(&point_slice);

        if remaining_points.is_empty() {
            return Ok((PyArray1::from_vec(py, Vec::new()), distance_to_projection));
        }

        let remaining = Polyline {
            points: remaining_points,
            dimension: self.dimension,
        };

        let cumulative_from_projection = remaining.arc_lengths_impl();
        Ok((
            PyArray1::from_vec(py, cumulative_from_projection),
            distance_to_projection,
        ))
    }

    /// Return a new polyline with the z coordinate set to zero (3D only).
    fn zero_out_z(&self) -> PyResult<Self> {
        if self.dimension != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "zero_out_z is only defined for 3D polylines",
            ));
        }

        let mut new_points = self.points.clone();
        for i in 0..self.len() {
            new_points[i * 3 + 2] = 0.0;
        }

        Ok(Self {
            points: new_points,
            dimension: 3,
        })
    }

    /// Apply a rigid transform to the waypoints (3D only).
    fn transform(&self, transform_pose: &Pose) -> PyResult<Self> {
        if self.dimension != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "transform is only defined for 3D polylines",
            ));
        }

        // Get the rotation matrix and translation from the pose
        let rot_mat = transform_pose.rotation_matrix();
        let pos = transform_pose.position();
        let translation: [f32; 3] = [pos.x, pos.y, pos.z];

        // Transform each point: rotated = R @ point + t
        let mut new_points = Vec::with_capacity(self.points.len());
        for i in 0..self.len() {
            let p = self.get_point(i);
            let x = rot_mat[0][0] * p[0] + rot_mat[0][1] * p[1] + rot_mat[0][2] * p[2] + translation[0];
            let y = rot_mat[1][0] * p[0] + rot_mat[1][1] * p[1] + rot_mat[1][2] * p[2] + translation[1];
            let z = rot_mat[2][0] * p[0] + rot_mat[2][1] * p[1] + rot_mat[2][2] * p[2] + translation[2];
            new_points.push(x);
            new_points.push(y);
            new_points.push(z);
        }

        Ok(Self {
            points: new_points,
            dimension: 3,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_lengths() {
        let polyline = Polyline {
            points: vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            dimension: 2,
        };

        let lengths = polyline.segment_lengths_impl();
        assert_eq!(lengths.len(), 2);
        assert!((lengths[0] - 1.0).abs() < 1e-10);
        assert!((lengths[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_arc_lengths() {
        let polyline = Polyline {
            points: vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0],
            dimension: 2,
        };

        let arc = polyline.arc_lengths_impl();
        assert_eq!(arc.len(), 3);
        assert!((arc[0] - 0.0).abs() < 1e-10);
        assert!((arc[1] - 1.0).abs() < 1e-10);
        assert!((arc[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_project_point() {
        let polyline = Polyline {
            points: vec![0.0, 0.0, 2.0, 0.0],
            dimension: 2,
        };

        // Project a point above the segment
        let result = polyline.project_point_impl(&[1.0, 1.0]);
        assert_eq!(result.segment_idx, 0);
        assert!((result.point[0] - 1.0).abs() < 1e-10);
        assert!((result.point[1] - 0.0).abs() < 1e-10);
        assert!((result.distance_along - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_3d_polyline() {
        let polyline = Polyline {
            points: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            dimension: 3,
        };

        assert_eq!(polyline.len(), 3);
        let lengths = polyline.segment_lengths_impl();
        assert_eq!(lengths.len(), 2);
        assert!((lengths[0] - 1.0).abs() < 1e-10);
        assert!((lengths[1] - 1.0).abs() < 1e-10);
    }
}
