# Alpasim Controller

This directory contains the Alpasim controller, which models vehicle dynamics and control.


## Testing

To run the tests, execute the following command from the project directory (`<repo_root>/src/controller`):

```bash
uv run pytest
```

## Benchmarking

Compare controller performance:

```bash
# Run benchmark with default ControllerConfig (linear MPC)
uv run -m benchmark run --output results/linear.json

# Run benchmark with a custom config (write a small YAML that sets ControllerConfig fields; see benchmark/README.md)
uv run -m benchmark run --output results/nonlinear.json --config path/to/controller.yaml

# Compare results
uv run -m benchmark compare results/nonlinear.json results/linear.json
```

## Implementation Notes

The controller receives trajectory reference commands, which are possibly delayed, along with state information
from the vehicle, which is used to constrain to the road surface. This information is forwarded by a
`SystemManager` to a `System` (vehicle dynamics + controller) which uses an MPC to compute commanded steering and
acceleration for the vehicle model, which then propagates the dynamics to the requested time and returns the new
state.

### Vehicle Model

The vehicle model is implemented as a planar dynamic bicycle model. The equations of motion can be found in
e.g. _Vehicle Dynamics and Control_ by Rajamani, with minor deviations to support the rear-axis coordinate
system definition. One dynamic model is used from standstill through highway speeds.

The lateral state `z = [vy_cg, yaw_rate]` is advanced with backward Euler at substeps of at most
10 ms. Multiplying the update by the end-of-step forward speed `v` gives
`(v I - h v A(v)) z_next = v z + h v b steering_next`. The matrix `v A(v)` has no division
by speed. At zero speed the tire matrix enforces zero lateral velocity and yaw rate; as speed
approaches zero, the steady solution approaches the small-angle no-slip kinematic limit.
There is no speed threshold, speed floor, or kinematic/dynamic blend in this update.

Steering and acceleration actuators use their exact constant-command solution. Position uses
trapezoidal velocities and midpoint heading, with `vy_rig = vy_cg - l_rig_to_cg * yaw_rate`
at all speeds. Reported body-velocity derivatives are differences over the final integration
substep, including the forward-speed clamp when stopping.

The simulator, linear MPC, and SQP MPC share the discrete update. At the zero-speed clamp,
symbolic predictors use the forward derivative. Nonlinear MPC retains implicit Radau
collocation of continuous dynamics. It uses the same tire coefficients with a smooth
regularization speed `sqrt(v**2 + 0.01**2)` and scales the steering force by
`v / sqrt(v**2 + 0.01**2)`. This preserves the steady-turn and no-slip limits without a
switch or standstill steering force, while avoiding a hard stop clamp inside IPOPT.
Its continuous predictor retains the existing small reverse-velocity slack; the simulator
still clamps forward speed at zero. Predictor agreement is tested separately from exact
parity of the discrete backends.

The model remains
forward-only and assumes linear tires and small steering/slip angles. Numerical stability at
low speed does not supply a standstill friction model, tire saturation, or reverse dynamics.
The speed-range tests cover 0–45 m/s, with numerical stress cases through 80 m/s; these are
numerical checks, not real-vehicle calibration.

### State Vector

The MPC and vehicle model assume a lateral/longitudinal decoupled system with state:

| Index | Name | Description |
|-------|------|-------------|
| 0 | `x` | x position of rig origin in inertial frame |
| 1 | `y` | y position of rig origin in inertial frame |
| 2 | `yaw` | yaw angle of rig origin in inertial frame |
| 3 | `vx_cg` | x component of CG velocity in body frame |
| 4 | `vy_cg` | y component of CG velocity in body frame |
| 5 | `yaw_rate` | yaw rate in body frame |
| 6 | `steering` | front wheel steering angle |
| 7 | `accel` | longitudinal acceleration state |

### Coordinate Frames

As the vehicle dynamics assume planar motion, additional frame constructions/transformations are required.
The controller/vehicle model introduces an `inertial frame`: a temporary reference frame that is
coincident/aligned with the vehicle `rig` frame at each time step. This frame allows for relative (planar)
motion to be computed and then added to the `local` to `rig` transformation.

### Step Execution

For each time step, the system will:
1. Override the current vehicle state (`local` to `rig` transformation and optionally the velocities)
2. "Drop" a new reference frame whose origin is coincident/aligned with the vehicle
3. Reset the initial state of the MPC based on the current vehicle state
4. Transform reference trajectory to rig frame
5. Run the MPC controller to compute commanded steering and acceleration
6. Propagate the vehicle model to the requested time
7. Apply the relative motion to the `local` to `rig` transformation

## Backend Implementations

Three MPC implementations are provided:

- `linear`: Linearizes the shared discrete bicycle model and solves an affine QP with `OSQP`.
- `nonlinear`: Uses smooth continuous bicycle dynamics with `do_mpc`/IPOPT.
- `sqp`: Repeatedly linearizes the nonlinear dynamics around the current SQP iterate, solves
  QP subproblems with `OSQP`, and regularizes changes between SQP iterates.

The implementations share the controller API and the common tracking/smoothness terms where
applicable. The linear controller prioritizes runtime, nonlinear MPC optimizes the full nonlinear
problem directly, and SQP is intended to recover most of the nonlinear tracking quality at lower
runtime by warm-starting from the reference trajectory and previous solve.

The desired implementation can be selected via the wizard controller config group
(e.g. `controller=nonlinear`) or by passing a YAML config via `--config` when running the server standalone.

`System` owns physical coercion: refresh derivatives, align held actuator commands,
and call the backend's required `reset_warm_start()`, including between MPC ticks.
`MPCController.compute_control()` owns command acceptance and failure cleanup;
backends implement `_compute_control()` and return a command/status or `None`/reason.
Unsuccessful or nonfinite solves return no command, a `failed:` status and cleared
warm starts. `System` retains its active command, which coercion synchronizes with
the supplied actuators. SQP rejects any failed QP; its outer iteration limit remains an
explicitly approximate accepted update. Holding a command does not guarantee safe driving.
Other controller implementations can use the same service interface.

Cold nonlinear solves seed their initial guess and previous-command penalty from the
supplied state and actuators; accepted subsequent solves retain the previous command.
Controller responses include per-request `solver_stats` with MPC update and failed
update counts, which runtime records in ASL. Held propagation and inner SQP iterations
do not count as updates. An absent field means unsupported diagnostics, rather than
zero updates.

### MPC Penalty Design

The MPC uses a quadratic penalty on the longitudinal position error, lateral position error, heading
error, and acceleration, as well as regularization terms on the relative changes of steering angle
commands and acceleration commands. The default time horizon is 2 sec (20 steps at 0.1s),
configurable via `controller.n_horizon` and `controller.dt_mpc`. There is a term that specifies at
which index along the horizon costs should start accumulating (to avoid over-penalizing initial
transients).

$$
J = \sum_{i=i_0}^{N} (w_{lon} e_{lon,i}^2 + w_{lat} e_{lat,i}^2 + w_{head} e_{head,i}^2 + w_{accel} a_i^2) + \sum_{i=1}^{N} (w_{\Delta steer} \Delta \delta_i^2 + w_{\Delta accel} \Delta a_i^2)
$$

### Nonlinear MPC

The formulation uses `do_mpc` to minimize the cost using the full nonlinear dynamic model. The
dynamics and cost function are defined symbolically using `casadi`, and the resulting nonlinear
program is solved using the IPOPT solver.

### Linear MPC

The linear MPC implementation casts the problem as a reduced form quadratic program (QP).
CasADi differentiates the shared discrete update about the current state and zero command.
The affine residual preserves the prediction at that operating point, including products of
speed and steering. The resulting state transition matrices and accumulated residual are
used to construct the horizon cost and constraints. The QP is solved using `OSQP`.

### SQP MPC

The SQP implementation builds a full-state reference from the reference poses: speed, curvature
and acceleration are derived from the poses; the velocity, yaw-rate, steering, and acceleration references are
rolled out through the bicycle model, and the position and heading references are
the reference poses themselves (the rollout carries rear-axle slip that the path-tangent poses do
not, so re-integrated positions would drift outside of turns). Each SQP iteration linearizes the nonlinear
dynamic bicycle model around the current state/control trajectory, solves a constrained QP with
`OSQP`, and stops when the step size falls below `gains.sqp_step_tolerance` or
`gains.sqp_max_iterations` is reached.

Additional SQP terms include velocity, lateral velocity, yaw-rate, and steering-state tracking, plus
proximal penalties `gains.sqp_state_step_weight` and `gains.sqp_control_step_weight` that stabilize
successive SQP iterates. `gains.max_curvature_abs` and `gains.max_curvature_speed_floor` define the
hard yaw-rate/curvature envelope used by both nonlinear and SQP MPC at low speed.

## Third-Party Licenses

This project uses [do_mpc](https://github.com/do-mpc/do-mpc) and [casadi](https://github.com/casadi/casadi/), which
are both licensed under the GNU General Public License v3.0 (GPL-3.0).

The linear MPC implementation uses [OSQP](https://osqp.org/), which is licensed under the Apache License 2.0.
