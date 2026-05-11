# AlpaSim MTGS Sensorsim

This package is a deployment wrapper for using MTGS as an AlpaSim
`SensorsimService`.

It intentionally does not vendor the full MTGS repository or checkpoint into
AlpaSim. The container image must provide the MTGS Python package and its CUDA /
Nerfstudio dependencies, while AlpaSim provides the gRPC API and wizard service
configuration.

Typical container paths:

- `/repo/src/mtgs_sensorsim`: this wrapper
- `/repo/src/grpc`: AlpaSim gRPC Python package
- `/mnt/mtgs`: MTGS repository, experiments, exported AlpaSim artifact directory

