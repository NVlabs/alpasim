# Alpasim Utils

This module contains utility functions for Alpamayo Sim that are shared across multiple services.

## Components

- **artifact.py**: Artifact management and loading utilities
- **trajectory.py**: Trajectory data structures and operations
- **pose.py**: Pose (position + quaternion) data structures
- **logs.py**: ASL log reading and writing utilities
- **scenario.py**: Scenario data structures (AABB, TrafficObjects, Rig, VehicleConfig, CameraId)
- **asl_to_frames/**: Command-line tool for extracting frames from ASL logs
- **print_asl/**: Command-line tool for printing ASL log contents

The core types (`Pose`, `Trajectory`, `Polyline`) are implemented in Rust via [utils_rs](../utils_rs/). See [`utils_rs.pyi`](../utils_rs/utils_rs.pyi) for the full list of available methods and their signatures.

## Installation

This module is typically installed as a dependency by other Alpasim services. It requires `alpasim_grpc` for protobuf message definitions.

## Usage

```python
from alpasim_utils.geometry import Pose, Trajectory, pose_from_grpc
from alpasim_utils.artifact import Artifact
from alpasim_utils.logs import async_read_pb_log
```
