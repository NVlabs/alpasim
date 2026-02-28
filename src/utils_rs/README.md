# utils_rs

Rust acceleration for `alpasim_utils`. Provides high-performance implementations of core data structures used in trajectory and pose manipulation.

## Components

- **Pose**: Single rigid transform (position + quaternion rotation)
- **Trajectory**: Timestamped trajectory with efficient interpolation and incremental updates
- **Polyline**: Efficient polyline storage with projection and interpolation

## Building

The extension is built with [maturin](https://github.com/PyO3/maturin). To rebuild after making changes:

```bash
uv pip install -e src/utils_rs --force-reinstall
```

This compiles the Rust code and installs the Python extension in editable mode.

## Usage

```python
import numpy as np
from utils_rs import Pose, Trajectory

# Pose for single rigid transforms
pose = Pose(np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

# Trajectory for timestamped pose sequences
timestamps = np.array([0, 100, 200], dtype=np.uint64)
positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32)
quaternions = np.array([[0.0, 0.0, 0.0, 1.0]] * 3, dtype=np.float32)
traj = Trajectory(timestamps, positions, quaternions)

# Interpolate to target timestamps
target_ts = np.array([50, 150], dtype=np.uint64)
interp_traj = traj.interpolate(target_ts)
interpolated_positions = interp_traj.positions  # returns (N, 3) array
```

## Development

Run Rust tests:

```bash
cd src/utils_rs
cargo test
```
