# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import numpy as np
from alpasim_grpc.v0.common_pb2 import AABB, Pose, Quat, Vec3
from alpasim_grpc.v0.physics_pb2 import PhysicsGroundIntersectionReturn
from scipy.spatial.transform import Rotation as R


def batch_so3_trans_2_se3(
    so3: np.ndarray = np.eye(3), trans: np.ndarray = np.zeros((3,))
) -> np.ndarray:
    assert (so3_dim := len(so3.shape)) == 3 or (trans_dim := len(trans.shape)) == 2

    if so3_dim == 3:
        batch_dim = so3.shape[0]
    if trans_dim == 2:
        batch_dim = trans.shape[0]

    if so3_dim == 2:
        so3 = np.repeat(so3[None, :, :], batch_dim, axis=0)
    elif trans_dim == 1:
        trans = np.repeat(trans[None, :], batch_dim, axis=0)

    batch_se3 = np.stack(
        [so3_trans_2_se3(so3=so3[i], trans=trans[i]) for i in range(batch_dim)]
    )
    return batch_se3


# TODO Add unittest to circle from one rep and back
def ndarray_to_vec3(array: np.ndarray) -> Vec3:
    assert array.shape == (3,)
    return Vec3(x=array[0], y=array[1], z=array[2])


def ndarray_to_quat(array: np.ndarray) -> Quat:
    assert array.shape == (4,)
    return Quat(w=array[0], x=array[1], y=array[2], z=array[3])


def scipy_to_quat(array: np.ndarray) -> Quat:
    assert array.shape == (4,)
    return Quat(x=array[0], y=array[1], z=array[2], w=array[3])


def ndarray_to_aabb(array: np.ndarray) -> AABB:
    assert array.shape == (3,)
    return AABB(size_x=array[0], size_y=array[1], size_z=array[2])


def aabb_to_ndarray(aabb: AABB) -> np.ndarray:
    return np.array([aabb.size_x, aabb.size_y, aabb.size_z])


def ndarray_to_pose(vec: np.ndarray, quat: np.ndarray) -> Pose:
    return Pose(vec=ndarray_to_vec3(vec), quat=ndarray_to_quat(quat))


def pose_status_to_grpc(
    pose: np.ndarray, status
) -> PhysicsGroundIntersectionReturn.ReturnPose:
    assert pose.shape == (4, 4)
    # scipy quat format (x, y, z, w)
    quat = R.from_matrix(pose[:3, :3]).as_quat(canonical=False)
    return PhysicsGroundIntersectionReturn.ReturnPose(
        pose=Pose(vec=ndarray_to_vec3(pose[:3, 3]), quat=scipy_to_quat(quat)),
        status=status.to_grpc(),
    )


def pose_grpc_to_ndarray(pose: Pose) -> np.ndarray:
    # scipy quat format (x, y, z, w)
    rot = R.from_quat([pose.quat.x, pose.quat.y, pose.quat.z, pose.quat.w]).as_matrix()
    vec = np.array([pose.vec.x, pose.vec.y, pose.vec.z])

    ndarray_pose = np.eye(4)
    ndarray_pose[:3, :3] = rot
    ndarray_pose[:3, 3] = vec
    return ndarray_pose


"""
These are borrowed from NCORE, see
https://gitlab-master.nvidia.com/Toronto_DL_Lab/ncore/-/blob/main/ncore/impl/common/transformations.py?ref_type=heads
"""


def so3_trans_2_se3(so3, trans):
    """Create a 4x4 rigid transformation matrix given so3 rotation and translation.

    Args:
        so3: rotation matrix [n,3,3]
        trans: x, y, z translation [n, 3]

    Returns:
        np.ndarray: the constructed transformation matrix [n,4,4]
    """

    if so3.ndim > 2:
        T = np.eye(4)
        T = np.tile(T, (so3.shape[0], 1, 1))
        T[:, 0:3, 0:3] = so3
        T[:, 0:3, 3] = trans.reshape(
            -1,
            3,
        )

    else:
        T = np.eye(4)
        T[0:3, 0:3] = so3
        T[0:3, 3] = trans.reshape(
            3,
        )

    return T


def euler_trans_2_se3(euler_angles, trans, degrees=True, seq="xyz"):
    """Create a 4x4 rigid transformation matrix given euler angles and translation.

    Args:
        euler_angles (np.array): euler angles [n,3]
        trans (Sequence[float]): x, y, z translation.
        seq string: sequence in which the euler angles are given

    Returns:
        np.ndarray: the constructed transformation matrix.
    """

    return so3_trans_2_se3(euler_2_so3(euler_angles, degrees), trans)


def euler_2_so3(euler_angles, degrees=True, seq="xyz"):
    """Converts the euler angles representation to the so3 rotation matrix
    Args:
        euler_angles (np.array): euler angles [n,3]
        degrees bool: True if angle is given in degrees else False
        seq string: sequence in which the euler angles are given

    Out:
        (np array): rotations given so3 matrix representation [n,3,3]
    """

    return (
        R.from_euler(seq=seq, angles=euler_angles, degrees=degrees)
        .as_matrix()
        .astype(np.float32)
    )


def transform_point_cloud(pc, T):
    """Transform the point cloud with the provided transformation matrix,
        support torch.Tensor and np.ndarry.
    Args:
        pc (np.array): point cloud coordinates (x,y,z) [num_pts, 3] or [bs, num_pts, 3]
        T (np.array): se3 transformation matrix  [4, 4] or [bs, 4, 4]

    Out:
        (np array): transformed point cloud coordinated [num_pts, 3] or [bs, num_pts, 3]
    """
    if len(pc.shape) == 3:
        if isinstance(pc, np.ndarray):
            trans_pts = T[:, :3, :3] @ pc.transpose(0, 2, 1) + T[:, :3, 3:4]
            return trans_pts.transpose(0, 2, 1)
        else:
            trans_pts = T[:, :3, :3] @ pc.permute(0, 2, 1) + T[:, :3, 3:4]
            return trans_pts.permute(0, 2, 1)

    else:
        trans_pts = T[:3, :3] @ pc.transpose() + T[:3, 3:4]
        return trans_pts.transpose()


def bbox_pose(bbox: np.ndarray) -> np.ndarray:
    """Converts an array-encoded bounding-box into a corresponding pose"""

    return np.block(
        [
            [
                R.from_euler("xyz", bbox[6:9], degrees=False).as_matrix(),
                np.array(bbox[:3]).reshape((3, 1)),
            ],
            [np.array([0, 0, 0, 1])],
        ]
    )


def pose_bbox(pose: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
    """Converts a pose with extents to an array-encoded bounding-box"""

    bbox = np.empty(9, dtype=dimensions.dtype)
    bbox[:3] = pose[:3, 3]  # centroid
    bbox[3:6] = dimensions  # dimensions from input
    bbox[6:9] = R.from_matrix(pose[:3, :3]).as_euler(
        "xyz", degrees=False
    )  # orientation

    return bbox


def transform_bbox(bbox_source: np.ndarray, T_source_target: np.ndarray) -> np.ndarray:
    """Applies a rigid-transformation to a bounding box
    Args:
       bbox (np.ndarray): bounding-box in source-frame parameterized by
            [x, y, z, length, width, height, eulerX, eulerY, eulerZ]
       T (np.array): se3 source->target transformation matrix to apply [4,4]
    Out:
       (np array): transformed bounding-box [num_pts, 3] or [bs, num_pts, 3]
    """

    # Convert bbox to corresponding pose
    T_bbox_source = bbox_pose(bbox_source)

    # Apply transformation
    T_bbox_target = T_source_target @ T_bbox_source

    # Convert back to bbox parametrization (dimensions stay unchanged)
    return pose_bbox(T_bbox_target, bbox_source[3:6])


def is_within_3d_bbox(pc: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Checks whether points of a point-cloud are in a 3d box

    [Reference implementation, consider faster C++ version in 'isWithin3DBBox']

    Args:
        pc: [N, 3] tensor. Inner dims are: [x, y, z].
        bbox: [9,] tensor. Inner dims are: [center_x, center_y, center_z, length, width, height, roll, pitch, yaw].
                        roll/pitch/yaw are in radians.
    Returns:
        point_in_bbox; [N,] boolean array.
    """

    center = bbox[0:3]
    dim = bbox[3:6]
    rotation_angles = bbox[6:9]

    # Get the rotation matrix from the heading angle
    rotation = R.from_euler("xyz", rotation_angles, degrees=False).as_matrix()

    # [4, 4]
    transform = so3_trans_2_se3(rotation, center)
    # [4, 4]
    transform = np.linalg.inv(transform)
    # [3, 3]
    rotation = transform[0:3, 0:3]
    # [3]
    translation = transform[0:3, 3]

    # [M, 3]
    points_in_box_frames = np.matmul(rotation, pc.transpose()).transpose() + translation

    # [M, 3]
    point_in_bbox = np.logical_and(
        np.logical_and(
            points_in_box_frames <= dim * 0.5, points_in_box_frames >= -dim * 0.5
        ),
        np.all(np.not_equal(dim, 0), axis=-1, keepdims=True),
    )

    # [N]
    point_in_bbox = np.prod(point_in_bbox, axis=-1).astype(bool)

    return point_in_bbox


def is_within_3d_bboxes(pc: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    """
    Wrapper for is_within_3d_bbox to iterate on multiple boxes to be consistent
    with C++ version 'isWithin3DBBox' in av_utils

    Args:
        pc: [N, 3] tensor. Inner dims are: [x, y, z].
        bboxes: [M, 9] tensor. Inner dims are: [center_x, center_y, center_z, length, width, height, roll, pitch, yaw].
                        roll/pitch/yaw are in radians.
    Returns:
        point_in_bboxes; [N,M] boolean array.
    """

    assert pc.shape[1] == 3, "Wrong PC input size"
    assert len(bboxes.shape) == 2, "bboxes need to be a 2D numpy array"
    assert bboxes.shape[1] == 9, "bboxes need to be a 2D numpy array"

    point_in_box = np.empty((pc.shape[0], bboxes.shape[0]), dtype=np.bool_)

    for i, bbox in enumerate(bboxes):
        point_in_box[:, i] = is_within_3d_bbox(pc, bbox)

    return point_in_box


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """Computed the inverse of a rigid transformation
    Args:
        T (np.array): se3 transformation matrix to invert [4,4]

    Out:
        (np array): inverse transformation [4,4]
    """
    Rt = T[:3, :3].transpose()
    return np.block([[Rt, -Rt @ T[:3, 3:]], [np.zeros((1, 3)), 1]])
