# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import math

import numpy as np
import pytest
from alpasim_grpc.v0.common_pb2 import AABB, Pose, Quat, Vec3
from alpasim_physics.backend import PhysicsBackend
from alpasim_physics.ply_io import save_mesh_vf
from alpasim_physics.utils import aabb_to_ndarray, pose_grpc_to_ndarray


@pytest.fixture
def default_aabb():
    aabb_grpc = AABB(size_x=5.0, size_y=2.0, size_z=1.0)
    return aabb_to_ndarray(aabb_grpc)


def generate_planar_mesh(z_offset: float) -> bytes:
    # Define the range and resolution of the grid
    x_min, x_max = -20.0, 20.0  # X range
    y_min, y_max = -20.0, 20.0  # Y range
    resolution = 0.2  # Grid spacing

    # Generate grid points
    x = np.arange(x_min, x_max + resolution, resolution)
    y = np.arange(y_min, y_max + resolution, resolution)
    x, y = np.meshgrid(x, y)
    z = z_offset * np.ones_like(x)

    # Flatten the grid to get point coordinates
    vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    # Generate triangular faces for the mesh
    num_rows, num_cols = x.shape
    faces = []

    for i in range(num_rows - 1):
        for j in range(num_cols - 1):
            # Indices of the vertices in the grid
            idx0 = i * num_cols + j
            idx1 = idx0 + 1
            idx2 = idx0 + num_cols
            idx3 = idx2 + 1

            # Create two triangles for the quad
            faces.append([idx0, idx1, idx2])  # Triangle 1
            faces.append([idx1, idx3, idx2])  # Triangle 2

    faces = np.array(faces)

    return save_mesh_vf(vertices, faces)


@pytest.mark.parametrize("z_offset", [0.0, 0.5, -0.25])
def test_update_pose_vertical_updates_on_planar_mesh(default_aabb, z_offset):
    mesh_ply = generate_planar_mesh(z_offset)
    physics = PhysicsBackend(env_mesh_ply=mesh_ply)

    POSE_Z_OFFSET = 0.3

    pose_z = default_aabb[2] / 2.0 + POSE_Z_OFFSET
    predicted_pose_grpc = Pose(
        vec=Vec3(x=3.0, y=0.0, z=pose_z),
        quat=Quat(w=1.0, x=0.0, y=0.0, z=0.0),
    )
    predicted_pose = pose_grpc_to_ndarray(predicted_pose_grpc)

    updated_pose, ground_intersection_status = physics.update_pose(
        predicted_pose,
        default_aabb,
        0,  # timestamp only used for visualization
    )

    translation_correction = updated_pose[:3, 3] - predicted_pose[:3, 3]
    assert translation_correction[2] == pytest.approx(z_offset - POSE_Z_OFFSET)


@pytest.mark.parametrize(
    "roll, pitch, pose_z_offset", [(0.01, 0.02, 0.1), (-0.015, 0.03, -0.1)]
)
def test_update_pose_trans_and_rot_updates_on_planar_mesh(
    default_aabb, roll, pitch, pose_z_offset
):
    mesh_ply = generate_planar_mesh(0.0)
    physics = PhysicsBackend(env_mesh_ply=mesh_ply)

    # create a pose that is slightly rotated and offset from the ground
    pose_z = default_aabb[2] / 2.0 + pose_z_offset
    qx = roll / 2.0
    qy = pitch / 2.0
    predicted_pose_grpc = Pose(
        vec=Vec3(x=3.0, y=0.0, z=pose_z),
        quat=Quat(x=qx, y=qy, z=0.0, w=math.sqrt(1.0 - qx**2 - qy**2)),
    )
    predicted_pose = pose_grpc_to_ndarray(predicted_pose_grpc)

    # Note: it appears that the full correction is not applied.
    # This may be a feature (so as not to over-correct on noise) or item
    # for further enhancements in the future. Either way, we apply the
    # correction multiple times to ensure that things settle
    for i in range(5):
        updated_pose, ground_intersection_status = physics.update_pose(
            predicted_pose,
            default_aabb,
            0,  # timestamp only used for visualization
        )
        predicted_pose = updated_pose

    EXPECTED_CONVERGED_POSE = pose_grpc_to_ndarray(
        Pose(
            vec=Vec3(x=3.0, y=0.0, z=default_aabb[2] / 2.0),
            quat=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
        )
    )

    assert updated_pose.flatten().tolist() == pytest.approx(
        EXPECTED_CONVERGED_POSE.flatten().tolist()
    )


def test_with_map():
    """
    This test is based on a problematic example seen in simulation
    where the ego vehicle bounced back and forth between two poses.
    The test attempts to prevent regressions of this sort by
    ensuring idempotency of the update_pose method for the
    troublesome pose.
    """
    physics = PhysicsBackend(
        env_mesh_ply=open("tests/data/mesh_ground.ply", "rb").read(),
    )
    aabb = aabb_to_ndarray(AABB(size_x=5.393, size_y=2.109, size_z=1.503))

    predicted_pose_grpc = Pose(
        vec=Vec3(
            x=1.88329315,
            y=-0.012246849,
            z=0.776931524,
        ),
        quat=Quat(
            x=-0.00160341803,
            y=-0.000724225305,
            z=-0.00506137963,
            w=0.999985635,
        ),
    )
    predicted_pose = pose_grpc_to_ndarray(predicted_pose_grpc)

    updated_pose, ground_intersection_status = physics.update_pose(
        predicted_pose,
        aabb,
        0,  # timestamp only used for visualization
    )
    double_updated_pose, ground_intersection_status = physics.update_pose(
        updated_pose,
        aabb,
        0,  # timestamp only used for visualization
    )

    assert (double_updated_pose - updated_pose).flatten().tolist() == pytest.approx(
        np.zeros(16).tolist(), abs=1e-4
    )
