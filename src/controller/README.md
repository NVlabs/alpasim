# Alpasim Controller
This directory contains the Alpasim controller, which models vehicle dynamics and control.

## Testing
To run the tests, execute the following command from the project directory (`<repo_root>/src/controller`):

```bash
uv run pytest
```

## Implementation Notes
The controller receives trajectory reference commands, which are possibly delayed, along with state information
from the vehicle, which is used to constrain to the road surface. This information is forwarded by a
`SystemManager` to a `System` (vehicle dynamics + controller) which uses an MPC to compute commanded steering and
acceleration for the vehicle model, which then propagates the dynamics to the requested time and returns the new
state.

[do_mpc](https://www.do-mpc.com/) is used for the MPC implementation, and the vehicle model is implemented in both a
symbolic representation (using `casadi` as required by `do_mpc`) and a non-symbolic implementation of the (mostly) same
dynamics. The equations of motion can be found in e.g. _Vehicle Dynamics and Control_ by Rajamani, with minor
deviations to support the rear-axis coordinate system definition. To avoid singularities at low speed, a kinematic model
is used below a speed threshold.

As the vehicle dynamics assume planar motion,
additional frame constructions/transformations are required. Namely, the controller/vehicle model introduces what is
referred to as the `inertial frame`: a temporary reference frame that is coincident/aligned with the vehicle `rig`
frame at some time step. This frame allows for relative (planar) motion to be computed and then added to the `local`
to `rig` transformation.

The MPC and vehicle model assume a lateral/longitudinal decoupled system with state:
* `x_inertial`:     x position of the rig origin relative to the inertial frame,
* `y_inertial`:     y position of the rig origin relative to the inertial frame,
* `yaw_inertial`:   yaw angle of the rig origin relative to the inertial frame,
* `body x-vel`:     x component of the cg velocity (relative to inertial frame),
                               resolved in the rig frame,
* `body y-vel`:     y component of the cg velocity (relative to inertial frame),
                               resolved in the rig frame,
* `yaw rate`:       yaw rate of the rig in the rig frame,
* `steering_angle`: steering angle of the front wheel,
* `acceleration`:   time derivative of the longitudinal velocity


For each time step, the system will:
* Override the current vehicle state (`local` to `rig` transformation and optionally the velocities)
* "Drop" a new reference frame whose origin is coincident/aligned with the vehicle
* Reset the initial state of the MPC based on the current vehicle state
* Run the MPC to compute the commanded steering and acceleration
* Propagate the vehicle model to the requested time using the commanded steering and acceleration
  to determine the relative motion of the vehicle
* Apply the relative motion to the `local` to `rig` transformation

### MPC Penalty Design
The MPC uses a quadratic penalty on the longitudinal position error, lateral position error, heading error, and
acceleration, as well as regularization terms on the relative changes of steering angle commands and acceleration
commands. The time horizon for the controller is 2 sec (20 steps at 0.1s), and there is a term that specifies at which
index along the horizon costs should start accumulating (to avoid over-penalizing initial transients).

$$
J = \sum_{i=i_0}^{N} (w_{lon} e_{lon,i}^2 + w_{lat} e_{lat,i}^2 + w_{head} e_{head,i}^2 + w_{accel} a_i^2) + \sum_{i=1}^{N} (w_{\Delta steer} \Delta \delta_i^2 + w_{\Delta accel} \Delta a_i^2)
$$

## Third-Party Licenses

This project uses [do_mpc](https://github.com/do-mpc/do-mpc) and [casadi](https://github.com/casadi/casadi/), which
are both licensed under the GNU Lesser General Public License v3.0.
