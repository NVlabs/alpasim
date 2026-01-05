# Alpasim Utils

This module contains utility functions for Alpamayo Sim that are shared across multiple services.

## Components

- **artifact.py**: Artifact management and loading utilities
- **trajectory.py**/**qvec.py**: Trajectory data structures and operations (QVec, Trajectory, DynamicState)
- **logs.py**: ASL log reading and writing utilities
- **scenario.py**: Scenario data structures (AABB, TrafficObjects, Rig, VehicleConfig, CameraId)
- **asl_to_frames/**: Command-line tool for extracting frames from ASL logs
- **print_asl/**: Command-line tool for printing ASL log contents

## Installation

This module is typically installed as a dependency by other Alpasim services. It requires `alpasim_grpc` for protobuf message definitions.

## Usage

```python
from alpasim_utils.qvec import QVec
from alpasim_utils.trajectory import Trajectory
from alpasim_utils.artifact import Artifact
from alpasim_utils.logs import async_read_pb_log
```

## Utilities

The package also includes some analysis utilities.

### ASL

Within `alpasim_utils`, there are tools for reading/writing alpasim logs (`.asl`) and converting
gRPC trajectory messages into python objects with useful methods.

There are also two executables that take advantage of these utilities:
* `uv run -m alpasim_utils.print_asl <args>` is useful for dumping the content of an `.asl` file
in human-readable string format for debugging purposes
* `uv run -m alpasim_utils.asl_to_frames` allows for dumping raw frames or `.mp4` videos which were
provided to the egodriver in the course of the simulation

In both cases use `--help` to learn about the command line arguments.
