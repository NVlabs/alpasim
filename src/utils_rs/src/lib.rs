// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 NVIDIA Corporation

//! utils_rs: Rust acceleration for alpasim_utils.
//!
//! This crate provides high-performance implementations of core data structures
//! used in trajectory and pose manipulation.

use pyo3::prelude::*;

mod array_utils;
mod polyline;
mod pose;
mod trajectory;

pub use polyline::Polyline;
pub use pose::Pose;
pub use trajectory::Trajectory;

/// Returns the version of this Rust extension.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// A Python module implemented in Rust.
#[pymodule]
fn utils_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Polyline>()?;
    m.add_class::<Pose>()?;
    m.add_class::<Trajectory>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
