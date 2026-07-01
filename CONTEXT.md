# AlpaSim

AlpaSim runs autonomous vehicle simulation experiments and records both driving
quality metrics and runtime telemetry for later analysis.

## Language

**Run UUID**:
The unique identifier for one AlpaSim startup/run directory. Use this to query
or inspect one concrete run instance.
_Avoid_: run id, run identifier

**Run Name**:
A human-readable label for an AlpaSim run. It is useful for display, but it is
not guaranteed to be unique.
_Avoid_: run id

**NRE Run ID**:
The upstream neural rendering/data-source run identifier attached to scene
metadata. It is distinct from an AlpaSim Run UUID.
_Avoid_: run id

**Runtime Parent**:
The top-level AlpaSim runtime owner for one startup. It owns run-level topology
and coordinates runtime workers.
_Avoid_: parent process, daemon, DaemonEngine

**Simulation Service**:
A runtime-facing service that participates in an AlpaSim simulation topology.
Simulation services are the services a Runtime Parent can connect to during a run.
_Avoid_: normal service

**Support Service**:
A wizard-managed service that supports an AlpaSim run without participating in
the simulation topology.
_Avoid_: normal service, telemetry service

**Prometheus Support Service**:
The Support Service that runs Prometheus-related observability processes for an
AlpaSim run.
_Avoid_: telemetry sidecar

**Slurm Process Exporter**:
The process metrics exporter used by the Prometheus Support Service on Slurm to
report CPU and memory usage for processes in one Slurm job.
_Avoid_: process telemetry sidecar
