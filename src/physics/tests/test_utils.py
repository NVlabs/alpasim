# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import numpy as np
import pytest
from alpasim_grpc.v0.common_pb2 import AABB, Pose, Quat, Vec3
from alpasim_physics.utils import (
    aabb_to_ndarray,
    ndarray_to_aabb,
    ndarray_to_pose,
    ndarray_to_quat,
    ndarray_to_vec3,
    pose_grpc_to_ndarray,
    pose_status_to_grpc,
    scipy_to_quat,
    se3_inverse,
    so3_trans_2_se3,
)
from scipy.spatial.transform import Rotation as R


class TestVec3Roundtrip:
    def test_ndarray_to_vec3_and_back(self):
        original = np.array([1.5, -2.3, 4.7])
        vec3 = ndarray_to_vec3(original)
        recovered = np.array([vec3.x, vec3.y, vec3.z])
        np.testing.assert_array_almost_equal(recovered, original)

    def test_ndarray_to_vec3_zeros(self):
        original = np.zeros(3)
        vec3 = ndarray_to_vec3(original)
        assert vec3.x == 0.0
        assert vec3.y == 0.0
        assert vec3.z == 0.0


class TestQuatRoundtrip:
    def test_ndarray_to_quat_and_back(self):
        # ndarray_to_quat uses [w, x, y, z] ordering
        original = np.array([0.5, 0.5, 0.5, 0.5])
        quat = ndarray_to_quat(original)
        recovered = np.array([quat.w, quat.x, quat.y, quat.z])
        np.testing.assert_array_almost_equal(recovered, original)

    def test_scipy_to_quat_and_back(self):
        # scipy_to_quat uses [x, y, z, w] ordering (scipy convention)
        original = np.array([0.5, 0.5, 0.5, 0.5])
        quat = scipy_to_quat(original)
        recovered = np.array([quat.x, quat.y, quat.z, quat.w])
        np.testing.assert_array_almost_equal(recovered, original)


class TestAABBRoundtrip:
    def test_ndarray_to_aabb_to_ndarray(self):
        original = np.array([5.0, 2.0, 1.5])
        aabb = ndarray_to_aabb(original)
        recovered = aabb_to_ndarray(aabb)
        np.testing.assert_array_almost_equal(recovered, original)


class TestPoseRoundtrip:
    def test_ndarray_to_pose_to_ndarray(self):
        """Verify that a pose survives the ndarray -> grpc -> ndarray roundtrip."""
        rotation = R.from_euler("xyz", [30, 45, 60], degrees=True)
        translation = np.array([10.0, -5.0, 3.0])
        se3 = so3_trans_2_se3(rotation.as_matrix(), translation)

        # Convert to grpc via ndarray_to_quat (w, x, y, z)
        quat_wxyz = np.array(
            [rotation.as_quat(canonical=False)[3], *rotation.as_quat(canonical=False)[:3]]
        )
        grpc_pose = ndarray_to_pose(translation, quat_wxyz)

        # Convert back
        recovered = pose_grpc_to_ndarray(grpc_pose)
        np.testing.assert_array_almost_equal(recovered[:3, 3], translation)
        np.testing.assert_array_almost_equal(recovered[:3, :3], se3[:3, :3], decimal=5)

    def test_identity_pose_roundtrip(self):
        identity_pose = Pose(
            vec=Vec3(x=0, y=0, z=0), quat=Quat(w=1, x=0, y=0, z=0)
        )
        recovered = pose_grpc_to_ndarray(identity_pose)
        np.testing.assert_array_almost_equal(recovered, np.eye(4))


class TestSE3Inverse:
    def test_inverse_is_identity(self):
        rotation = R.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
        translation = np.array([1.0, 2.0, 3.0])
        T = so3_trans_2_se3(rotation, translation)
        T_inv = se3_inverse(T)
        product = T @ T_inv
        np.testing.assert_array_almost_equal(product, np.eye(4))

    def test_inverse_of_identity_is_identity(self):
        T_inv = se3_inverse(np.eye(4))
        np.testing.assert_array_almost_equal(T_inv, np.eye(4))
