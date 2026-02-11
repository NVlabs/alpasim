# Alpasim design

Alpasim is an AV simulator focused on three principles:

1. Sensor fidelity
1. Horizontal scalability
1. Hackability for research

Real-time and very precise physics are non-goals.

For these reasons we implement Alpasim as a collection of microservices (enabling scalability) which
are implemented in Python (accessible to researchers) and communicate via gRPC.

Core services include the
[Neural Rendering Engine (NRE)](https://www.nvidia.com/en-us/glossary/3d-reconstruction/) and neural
traffic simulator (coming soon). Additionally, we have a [physics simulation module](/src/physics)
(ground constraints for egovehicle and non-ego actors), a
[controller/vehicle model](/src/controller), and a
[runtime](/src/runtime) which drives the simulation
loop by issuing calls to the respective services and produces logs. An [eval module](src/eval) runs
outside of the main simulation loop and consumes the logs to compute metrics for autonomous driving.

The simulator interfaces with [a driver](/src/driver) - the egovehicle policy network, which is the
main target of the simulation and creates trajectories to completee the feedback loop. The services
communicate with a gRPC protocol defined in the [gRPC API](/src/grpc/).

![Alpasim architecture diagram](/docs/assets/images/alpasim-architecture.png)

## Data flow of the simulation

The diagram illustrates the logical flow of the simulation instance

- **Wizard** sets up the simulation configuration and (often) launches the microservices
- **Runtime** keeps track of the world state
- The world state is fed as bounding boxes to **trafficsim**, which actuates non-ego actors
  (pedestrians, vehicles, etc)
- The world state is also used by **NRE** to produce camera frames for the ego vehicle
- Sensor readings are used by the **driver** to make decisions about actuating the ego vehicle
- The actuation request (planned path) is fed to the **controller**, which models the vehicle
  controller and dynamics, providing (uncorrected) egomotion
- Both the actuation of the ego vehicle and actors is passed to **physics** which applies
  constraints such that the vehicles stay on the ground
- The resulting updated state is passed to the **runtime** which logs it and repeats the loop
- The logs persist after simulation and can be used to compute metrics with the **eval** module

The software implementation places the runtime at the center, as a node for all communications. The
remaining services can be replicated according to their computational requirements (in general
`ego policy > sensor sim > controller sim > traffic sim > physics sim`). This design facilitates
synchronized logging and lets the runtime double as a load balancer between the replicas of
remaining services but it also means that the runtime is about as IO intense as all other services
_combined_.

The runtime is a gRPC client and needs to be aware of the addresses of all other microservices; the
microservices are server daemons and make no requests of their own. The containers can be run in any
way the user wishes (on arbitrary multiple machines) as long as the runtime is aware of the
addresses and the filesystem mounts contain the necessary files. This repository focuses on
configuration for running them all jointly on a single machine via `docker compose` or with `slurm`.

## Links to source code

The microservices/components can be found here:

- [controller](/src/controller): a simple vehicle controller + model
- [driver](/src/driver): a service that runs driving policies
- [eval](/src/eval): an evaluation framework that processes data
- [physics](/src/physics): ground-mesh interaction modeling
- [runtime](/src/runtime): the simulation runtime
- trafficsim (coming soon)
