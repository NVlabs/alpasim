# Alpasim Controller

This directory contains the Alpasim controller, which models vehicle dynamics and control.

## Testing

To run the tests, execute the following command from the project directory
(`<repo_root>/src/controller`):

```bash
uv run pytest
```

## Benchmarking

Compare controller performance:

```bash
# Run benchmark with linear controller
uv run -m benchmark run --output results/linear.json --backend=linear

# Run benchmark with nonlinear controller
uv run -m benchmark run --output results/nonlinear.json --backend=nonlinear

# Compare results
uv run -m benchmark compare results/nonlinear.json results/linear.json
```

## Implementation Notes

The controller receives trajectory reference commands, which are possibly delayed, along with state
information from the vehicle, which is used to constrain to the road surface. This information is
forwarded by a `SystemManager` to a `System` (vehicle dynamics + controller) which uses an MPC to
compute commanded steering and acceleration for the vehicle model, which then propagates the
dynamics to the requested time and returns the new state.

### Vehicle Model

The vehicle model is implemented as a planar dynamic bicycle model. The equations of motion can be
found in e.g. _Vehicle Dynamics and Control_ by Rajamani, with minor deviations to support the
rear-axis coordinate system definition. To avoid singularities at low speed, a kinematic model is
used below a speed threshold.

### State Vector

The MPC and vehicle model assume a lateral/longitudinal decoupled system with state:

| Index | Name       | Description                                |
| ----- | ---------- | ------------------------------------------ |
| 0     | `x`        | x position of rig origin in inertial frame |
| 1     | `y`        | y position of rig origin in inertial frame |
| 2     | `yaw`      | yaw angle of rig origin in inertial frame  |
| 3     | `vx_cg`    | x component of CG velocity in body frame   |
| 4     | `vy_cg`    | y component of CG velocity in body frame   |
| 5     | `yaw_rate` | yaw rate in body frame                     |
| 6     | `steering` | front wheel steering angle                 |
| 7     | `accel`    | longitudinal acceleration state            |

### Coordinate Frames

As the vehicle dynamics assume planar motion, additional frame constructions/transformations are
required. The controller/vehicle model introduces an `inertial frame`: a temporary reference frame
that is coincident/aligned with the vehicle `rig` frame at each time step. This frame allows for
relative (planar) motion to be computed and then added to the `local` to `rig` transformation.

### Step Execution

For each time step, the system will:

1. Override the current vehicle state (`local` to `rig` transformation and optionally the
   velocities)
1. "Drop" a new reference frame whose origin is coincident/aligned with the vehicle
1. Reset the initial state of the MPC based on the current vehicle state
1. Transform reference trajectory to rig frame
1. Run the MPC controller to compute commanded steering and acceleration
1. Propagate the vehicle model to the requested time
1. Apply the relative motion to the `local` to `rig` transformation

## Backend Implementations

Two MPC implementations are provided: a nonlinear MPC using `do_mpc` and a linear MPC using `OSQP`.
In both cases, the same cost function/constraints are used, but the problem formulation and solvers
differ. The trade-off between the two implementations is speed vs. accuracy--the responses are
similar for most normal driving scenarios, but the nonlinear MPC is more accurate for aggressive
maneuvers or tight turns.

The desired implementation can be selected via the `--mpc-implementation` flag when running the
service or the benchmark, with choices of `nonlinear` or `linear`.

### MPC Penalty Design

The MPC uses a quadratic penalty on the longitudinal position error, lateral position error, heading
error, and acceleration, as well as regularization terms on the relative changes of steering angle
commands and acceleration commands. The time horizon for the controller is 2 sec (20 steps at 0.1s),
and there is a term that specifies at which index along the horizon costs should start accumulating
(to avoid over-penalizing initial transients).

$$
J = \sum_{i=i_0}^{N} (w_{lon} e_{lon,i}^2 + w_{lat} e_{lat,i}^2 + w_{head} e_{head,i}^2 + w_{accel} a_i^2) + \sum_{i=1}^{N} (w_{\Delta steer} \Delta \delta_i^2 + w_{\Delta accel} \Delta a_i^2)
$$

### Nonlinear MPC

The formulation uses `do_mpc` to minimize the cost using the full nonlinear dynamic model. The
dynamics and cost function are defined symbolically using `casadi`, and the resulting nonlinear
program is solved using the IPOPT solver.

### Linear MPC

The linear MPC implementation casts the problem as a reduced form quadratic program (QP). Starting
from the quadratic cost function, the dynamics are linearized about the "free" trajectory (i.e.
assuming no control inputs) at each time step to for the linearized perturbation dynamics. After
discretizing, the full state transition matrices are constructed over the horizon, and the QP is
formed by plugging those dynamics into the cost function. The resulting QP is solved using `OSQP`.

## Third-Party Licenses

This project uses [do_mpc](https://github.com/do-mpc/do-mpc) and
[casadi](https://github.com/casadi/casadi/), which are both licensed under the GNU General Public
License v3.0 (GPL-3.0).

The linear MPC implementation uses [OSQP](https://osqp.org/), which is licensed under the Apache
License 2.0.
