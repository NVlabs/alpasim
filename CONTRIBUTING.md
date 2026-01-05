# Contributing to the AlpaSim Docker repository

This page outlines the guidelines for anybody who wishes to contribute to the AlpaSim repository.

## Release Process

There is no formal release process for the AlpaSim repository. Once CICD pipelines are enabled, we
will automate the versioning and release process, although there is no plan for formal
qualification/certification of releases at this time.

### (Future) Automatic Versioning

Services automatically receive minor version bumps when their code changes in merge requests. This
also includes rebuilding the docker container and uploading the corresponding sqash file.

> :warning: Until this is enabled, please manually bump versions in `pyproject.toml` files when
> making changes.

### Post-merge release pipeline

After a merge request is merged, the pipeline on main will check the following, for each package:

1. Was the code in this package changed in this commit?
1. If so, does a git tag already exist for this package with the same version number?
1. If not, this indicates a new tag will be created and the new version of the package should be
   published.

### Common challenges

**I have to rebase all the time! :(**

This repository is configured to require fast-forward merges only. This is to ensure the commit
history is linear and easy to reason about. This means if you are being main, you will be required
to rebase. This is easy, follow the steps below:

```sh
# Assuming you are checked out to your MR branch.
git fetch origin main
git rebase origin/main
# Be advised, the following command is a force push.
git push origin +HEAD
```

A force push is required because the history has been rewritten. Try and avoid force pushing if you
can, but it is necessary when rebasing.

## Coding Standards and Style Guides

Linting is performed using [pre-commit](https://pre-commit.com/) with
[black](https://black.readthedocs.io/en/stable/). To set up locally, first install `pre-commit`
(e.g. using `pip`) and then set up with `pre-commit install`. This will install the necessary
pre-commit hooks, which can be run manually with `pre-commit run --all-files`.

### Variables and Naming Conventions

Generally, one should follow the
[PEP8 guidelines for naming conventions.](https://peps.python.org/pep-0008/#naming-conventions).
However, due to the subject matter in the simulation, there are a few additional conventions that
help with readability and can prevent costly misunderstandings.

#### Coordinate Systems

There are four primary coordinate systems that are used in simulations:

1. The `local` frame: an inertial frame, fixed on a per-scenario basis, which represents an ENU
   frame defined by NRE.
1. The `rig` frame: a body-fixed frame which has the following properties:
   - The x-axis points forward
   - The y-axis points to the left when looking forward
   - The z-axis points up
   - The origin is at the center of the rear axle, projected onto the ground plane
1. The `aabb` (Axis-Aligned Bounding Box) frame: a body-fixed frame, defined with the same
   orientation as the rig definition, but the origin at the center of the object bounding box.
1. The `ecef` (Earth-Centered, Earth-Fixed) frame: an inertial global frame, based on WGS84.

Additionally, to mimic proprioceptive noise, the ego-position sent to the driver is in a `noised`,
or "estimated" frame (can either be thought of as a `local -> rig_est` transformation or,
equivalently, a `local_est -> rig` transformation). The runtime is responsible for translating the
returned waypoints back into the (unnoised) `local` frame.

To aid in readability/clarity, it is important to develop a consistent naming convention for
physical quantities that are defined in these frames.

#### On coordinate frames and transforms.

Throughout the code we will use transforms and also sometimes describe positions as transforms.

Definition: An active transform moves or rotates an object within a fixed coordinate frame, while a
passive transform changes the coordinate frame itself in which the object is described.

There's a correspondence between _transforms_ and _positions_: The position of `B` in the coordinate
frame of `A` is the same as the _active transform_ from `A` to `B`, i.e. `A->B`. In contrast, in
order to change the notation of a position from the coordinate frame `A` to coordinate frame `B`
(for some position) we need the _passive transform_ from A to B, which is the inverse of the active
transform, i.e. `B->A = (A->B).inverse()`!

#### Vectors, Rotations, and Poses

Vectors representing positions, velocities, and accelerations should be named according to the frame
in which they are defined. For example, the position of some location relative to another should
include enough information to determine the "tip" and "tail" of the vector as well as the frame in
which it is defined.

For example, to specify the position of some object in the local frame:

```python
# Bad
position = np.array([1, 2, 3]) # What location is specified? What frame is this in?
position_local = np.array([1, 2, 3]) # Better, but still unclear which object is being referenced

# Good
position_object_local = np.array([1, 2, 3])
# or, alternatives
position_object_in_local = np.array([1, 2, 3])
object_position_local = np.array([1, 2, 3])
```

To specify a relative position between two objects:

```python
# Bad
position_front_axle = np.array([3, 0.0, 0.1]) # reference frame unclear
obj1_to_obj2 = np.array([10.0, 20.0, 0.0]) # reference frame unclear

# Good
position_front_axle_in_rig = np.array([3, 0.0, 0.1]) # front_axle_in_rig also acceptable
obj1_to_obj2_in_aabb = np.array([10.0, 20.0, 0.0]) # relative position of obj1 to obj2 in aabb frame
```

Throughout the simulation codebase, unless otherwise specified, rotations and poses should always
follow an active convention: that is the "A to B" transform operated on some quantity should take
that quantity in frame A and "move" it using the "A to B" transform.

For example, to specify the rotation of some object relative to the local frame:

```python
# Bad
rotation = Quaternion(...), DirectionCosine(...), AxisAngle(...),  # What rotation is specified?
                                                                   # What frames are involved?
pose = Pose(...) # What pose is specified? What frames are involved?

# Good
rotation_local_to_rig = Quaternion(...), DirectionCosine(...), AxisAngle(...)
pose_rig_to_aabb = Pose(...)
transform_rig_to_aabb = Pose(...) # also fine
```

Bringing the ideas together, see an example usage of the above conventions:

```python
# Positions
rig_to_aabb_in_local = np.array(...)
position_rig_local = np.array(...)
position_aabb_local = position_rig_local + rig_to_aabb_in_local

# Poses
pose_ego_rig_to_aabb = QVec(vec3=..., quat=...)
pose_local_to_ego_rig = QVec(vec3=..., quat=...)

position_ego_aabb_local = (pose_local_to_ego_rig @ pose_ego_rig_to_aabb).vec3

```

#### Overview over which coordinate frames are used for communiation with services:

- **Driver service**

  - `submit_trajectory`: sends the noised history in the `local` frame (i.e. the noisy rig location
    in the local coordinate frame)
  - `submit_route`: sends waypoints in the `noisy rig` frame.
  - `submit_recording_ground_truth`: sends the gt trajectory in the `rig` frame. Note that because
    the driver doesn't know it's true location (only the estimated one), if it transforms the GT to
    the local frame, there's some error.
  - `drive` responses return poses in (what it thinks is the) `local` frame, but since its
    proprioception is noisy, these poses need to be mapped by runtime from a "noisy" `local` frame
    to the "true" `local` frame.

- **Controller/vehicle service** expects the current `pose_local_to_rig`, linear/angular velocities,
  and a `rig`-frame reference trajectory; responses return future `local->rig` poses (i.e. rig in
  local frame) and their estimates.

- **Physics service** expects ego and traffic poses as `local -> AABB` transformations and returns
  them in the same frame.

- **Traffic service** expects all communication as `local -> AABB` transformations.

- **Sensorsim service** expects the rig trajectory in `local` frame plus per-camera calibration
  `rig->sensor_pose`, and the `local->AABB` trajectories for rendering dynamic objects.

- **Logging:** `ActorPoses` log entries store every actor in the `AABB` frame (relative to the
  `local` frame), while metadata captures the `rig->AABB` transform for replay.

- **RoadCast:** Per DriveWorks conventions, the majority of quantities are reflected in the `rig`
  frame.

## Code Review

Merge/Pull requests are required for all changes to the codebase. Templates are provided to ensure
consistency and completeness.

## Code of Conduct

This project follows the
[NVIDIA Code of Conduct](https://images.nvidia.com/aem-dam/en-zz/Solutions/about-us/NVIDIA-Code-of-Conduct-External.pdf).
