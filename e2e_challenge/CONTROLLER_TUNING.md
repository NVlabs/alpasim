# Controller Tuning and Trajectory Validity

The challenge evaluates a trajectory-producing driver together with the
official nonlinear model predictive controller (MPC) and vehicle model. The
controller converts each reference trajectory into steering and acceleration
commands, and the resulting vehicle motion determines the observations the
driver receives next. The score therefore measures the coupled closed-loop
system, not the policy trajectory in isolation.

## Why Controller Gains Are Tunable

Many learned driving policies are trained on offline expert data. In closed
loop, however, a policy's earlier outputs affect the states and observations it
encounters later. Small prediction or tracking errors can therefore move the
vehicle away from the distribution represented in the training data and
compound over time. This sequential distribution-shift problem is a
well-established limitation of behavior cloning, for example.

For this reason, the challenge exposes a bounded subset of the nonlinear MPC's
cost-function gains. This lets contestants tune the policy and controller as a
coupled system while the controller implementation, vehicle dynamics,
constraints, and all other simulator settings remain fixed across submissions.

See the [challenge CLI documentation](competitor_cli/README.md#optional-nonlinear-controller-gains)
for the accepted ranges and submission command, and the
[starter-kit example](starter_kit/controller_gains.example.json) for the
default gain set.

## Expected Trajectory Domain

The official nonlinear MPC is a trajectory-tracking controller. It is not a
general trajectory validator, trajectory-repair system, or safety filter. It
expects reference trajectories that are sufficiently smooth and dynamically
plausible for the simulated vehicle. A syntactically valid trajectory is not
necessarily dynamically feasible, and the controller is not guaranteed to
respond gracefully to every finite-valued policy output.

In particular, we have occasionally observed a failure mode when a reference
trajectory requests extremely harsh, physically unrealistic braking. In this
case, the nonlinear optimization can produce a zig-zagging solution as it tries
to reduce longitudinal progress. The resulting steering response can be harsh,
and the vehicle may not recover during the rollout.

Robust recovery from arbitrary or dynamically unrealistic policy trajectories,
including reference repair and fallback control, is outside the scope of this
competition. Contestants should treat trajectory validity as part of their
system design. Controller-gain tuning may improve compatibility between a
policy and the official vehicle response, but it should not be relied upon to
make arbitrary trajectories safe or trackable.
