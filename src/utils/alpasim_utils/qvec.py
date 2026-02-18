# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Utility definitions for quaternion + translation vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

import numpy as np
from alpasim_grpc.v0 import common_pb2 as grpc_types
from scipy.spatial.transform import Rotation as R


def assert_is_quat_shape(q: np.ndarray) -> None:
    if q.shape[-1] != 4:
        raise ValueError(f"Expected last dimension to be 4, got {q.shape[-1]}")


def assert_is_vec3_shape(v: np.ndarray) -> None:
    if v.shape[-1] != 3:
        raise ValueError(f"Expected last dimension to be 3, got {v.shape[-1]}")


def quat_vec3_multiply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    assert_is_quat_shape(q)
    assert_is_vec3_shape(v)

    rotation = R.from_quat(q)

    zeros_shape = v.shape[:-1] + (1,)
    zeros = np.zeros(zeros_shape)
    v_as_r = R.from_quat(np.concatenate([v, zeros], axis=-1))

    result = rotation * v_as_r * rotation.inv()
    return result.as_quat()[..., :-1]


try:  # Python 3.11+
    from typing import Self
except ImportError:  # pragma: no cover - fallback for older versions
    from typing_extensions import Self


@dataclass(kw_only=True)
class QVec:
    """Quaternion + translation representation."""

    vec3: np.ndarray
    quat: np.ndarray  # scipy quat format (x, y, z, w)

    def __post_init__(self) -> None:
        assert_is_vec3_shape(self.vec3)
        assert_is_quat_shape(self.quat)
        assert self.vec3.shape[:-1] == self.quat.shape[:-1]

    def __repr__(self) -> str:
        return f"QVec(batch_size={self.batch_size})"

    def clone(self) -> Self:
        return QVec(vec3=self.vec3.copy(), quat=self.quat.copy())

    @classmethod
    def create_empty(cls) -> Self:
        return cls(
            vec3=np.zeros((0, 3), dtype=np.float32),
            quat=np.zeros((0, 4), dtype=np.float32),
        )

    @property
    def batch_size(self) -> tuple[int, ...]:
        return self.vec3.shape[:-1]

    @staticmethod
    def stack(qvecs: Sequence["QVec"], axis: int = 0) -> "QVec":
        return QVec(
            vec3=np.stack([qvec.vec3 for qvec in qvecs], axis=axis),
            quat=np.stack([qvec.quat for qvec in qvecs], axis=axis),
        )

    def _apply(self, fn: Callable[[np.ndarray], np.ndarray]) -> "QVec":
        return QVec(vec3=fn(self.vec3), quat=fn(self.quat))

    def __getitem__(self, index: Any) -> "QVec":
        return self._apply(lambda arr: arr[index])

    def __len__(self) -> int:
        return self.batch_size[0]

    def __iter__(self) -> Iterator["QVec"]:
        for i in range(len(self)):
            yield self[i]

    def __matmul__(self, other: Any) -> Any:
        rotation = R.from_quat(self.quat)
        if isinstance(other, QVec):
            vec3 = (rotation.as_matrix() @ other.vec3[..., None]).squeeze(
                -1
            ) + self.vec3
            quat = (rotation * R.from_quat(other.quat)).as_quat()
            return QVec(vec3=vec3, quat=quat)

        return NotImplemented

    def rotation_only(self) -> "QVec":
        return QVec(vec3=np.zeros_like(self.vec3), quat=self.quat)

    def inverse(self) -> "QVec":
        rotation = R.from_quat(self.quat).inv()
        return QVec(
            vec3=-(rotation.as_matrix() @ self.vec3),
            quat=rotation.as_quat(),
        )

    def as_se3(self) -> np.ndarray:
        m = np.zeros((*self.batch_size, 4, 4))
        m[..., 3, 3] = 1

        m[..., :3, :3] = R.from_quat(self.quat).as_matrix()
        m[..., :3, 3] = self.vec3[..., :3]

        return m

    @staticmethod
    def from_se3(se3_mat: np.ndarray) -> "QVec":
        quat = R.from_matrix(se3_mat[..., :3, :3]).as_quat()
        vec3 = se3_mat[..., :3, 3]
        return QVec(vec3=vec3, quat=quat)

    def append(self, other: "QVec") -> "QVec":
        first = self[None] if self.batch_size == () else self

        if other.batch_size == ():
            other = other[None]

        return QVec(
            vec3=np.concatenate([first.vec3, other.vec3], axis=0),
            quat=np.concatenate([first.quat, other.quat], axis=0),
        )

    def as_grpc_pose(self):  # type: ignore[override]
        from alpasim_grpc.v0 import common_pb2 as grpc_types

        assert self.batch_size == ()
        return grpc_types.Pose(
            vec=grpc_types.Vec3(
                x=self.vec3[0],
                y=self.vec3[1],
                z=self.vec3[2],
            ),
            quat=grpc_types.Quat(
                x=self.quat[0],
                y=self.quat[1],
                z=self.quat[2],
                w=self.quat[3],
            ),
        )

    def to_grpc_pose_at_time(self, timestamp_us: int):
        from alpasim_grpc.v0 import common_pb2 as grpc_types

        return grpc_types.PoseAtTime(
            pose=self.as_grpc_pose(),
            timestamp_us=timestamp_us,
        )

    def as_grpc_poses(self) -> Sequence[grpc_types.Pose]:
        assert len(self.batch_size) == 1
        return [qvec.as_grpc_pose() for qvec in self]

    @staticmethod
    def from_grpc_pose(grpc_pose: grpc_types.Pose) -> "QVec":
        return QVec(
            vec3=np.array([getattr(grpc_pose.vec, dim) for dim in "xyz"]),
            quat=np.array([getattr(grpc_pose.quat, dim) for dim in "xyzw"]),
        )

    @property
    def yaw(self) -> np.ndarray:
        return R.from_quat(self.quat).as_euler("xyz")[..., 2]
