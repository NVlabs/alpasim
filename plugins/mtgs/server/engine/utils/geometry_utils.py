# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Geometry utilities for MTGS rendering.

Includes the Sim2 class for 2D similarity transformations and
functions for height map queries and box pose computation.
"""

import json
import math
import numbers
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion


@dataclass(frozen=True)
class Sim2:
    """Similarity(2) lie group object.

    Args:
        R: array of shape (2x2) representing 2d rotation matrix
        t: array of shape (2,) representing 2d translation
        s: scaling factor
    """

    R: npt.NDArray[np.float64]
    t: npt.NDArray[np.float64]
    s: float

    def __post_init__(self) -> None:
        assert self.R.shape == (2, 2)
        assert self.t.shape == (2,)

        if not isinstance(self.s, numbers.Number):
            raise ValueError("Scale `s` must be a numeric type!")
        if math.isclose(self.s, 0.0):
            raise ZeroDivisionError(
                "3x3 matrix formation would require division by zero"
            )

    def __repr__(self) -> str:
        trans = np.round(self.t, 2)
        return (
            f"Angle (deg.): {self.theta_deg:.1f}, Trans.: {trans}, Scale: {self.s:.1f}"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sim2):
            return False
        if not np.isclose(self.scale, other.scale):
            return False
        if not np.allclose(self.rotation, other.rotation):
            return False
        if not np.allclose(self.translation, other.translation):
            return False
        return True

    @property
    def theta_deg(self) -> float:
        c, s = self.R[0, 0], self.R[1, 0]
        theta_rad = np.arctan2(s, c)
        return float(np.rad2deg(theta_rad))

    @property
    def rotation(self) -> npt.NDArray[np.float64]:
        return self.R

    @property
    def translation(self) -> npt.NDArray[np.float64]:
        return self.t

    @property
    def scale(self) -> float:
        return self.s

    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        T = np.zeros((3, 3))
        T[:2, :2] = self.R
        T[:2, 2] = self.t
        T[2, 2] = 1 / self.s
        return T

    def compose(self, S):
        return Sim2(
            R=self.R @ S.R,
            t=self.R @ S.t + ((1.0 / S.s) * self.t),
            s=self.s * S.s,
        )

    def inverse(self):
        Rt = self.R.T
        sRt = -Rt @ (self.s * self.t)
        return Sim2(Rt, sRt, 1.0 / self.s)

    def transform_from(self, points_xy):
        if not points_xy.ndim == 2:
            raise ValueError("Input points are not 2-dimensional.")
        assert points_xy.shape[1] == 2
        transformed_point_cloud = (points_xy @ self.R.T) + self.t
        return transformed_point_cloud * self.s

    @classmethod
    def from_json(cls, json_fpath: Path):
        with json_fpath.open("r") as f:
            json_data = json.load(f)
        R = np.array(json_data["R"]).reshape(2, 2)
        t = np.array(json_data["t"]).reshape(2)
        s = float(json_data["s"])
        return cls(R, t, s)

    @classmethod
    def from_matrix(cls, T: npt.NDArray[np.float64]):
        if np.isclose(T[2, 2], 0.0):
            raise ZeroDivisionError(
                "Sim(2) scale calculation would lead to division by zero."
            )
        R = T[:2, :2]
        t = T[:2, 2]
        s = 1 / T[2, 2]
        return cls(R, t, s)


def get_raster_values_at_coords(
    points_xy: np.ndarray, sim2: Sim2, np_image: np.ndarray, fill_value=np.nan
):
    city_coords = points_xy[:, :2]
    npyimage_coords = sim2.transform_from(city_coords)
    npyimage_coords = npyimage_coords.astype(np.int64)

    raster_values = np.full((npyimage_coords.shape[0]), fill_value)
    ind_valid_pts = (
        (npyimage_coords[:, 1] >= 0)
        * (npyimage_coords[:, 1] < np_image.shape[0])
        * (npyimage_coords[:, 0] >= 0)
        * (npyimage_coords[:, 0] < np_image.shape[1])
    )
    raster_values[ind_valid_pts] = np_image[
        npyimage_coords[ind_valid_pts, 1], npyimage_coords[ind_valid_pts, 0]
    ]
    return raster_values


def calculate_box_eular_angle_and_z(
    height_map: np.ndarray, sim2: Sim2, box: np.ndarray
):
    if box.ndim == 1:
        assert box.shape[0] == 6
        box = box.reshape(1, -1)
    else:
        assert box.ndim == 2 and box.shape[1] == 6

    x, y, length, w, h, yaw = box.T
    length = length / 1.8

    corners = np.array(
        [
            [
                x + length / 2 * np.cos(yaw) - w / 2 * np.sin(yaw),
                y + length / 2 * np.sin(yaw) + w / 2 * np.cos(yaw),
            ],
            [
                x + length / 2 * np.cos(yaw) + w / 2 * np.sin(yaw),
                y + length / 2 * np.sin(yaw) - w / 2 * np.cos(yaw),
            ],
            [
                x - length / 2 * np.cos(yaw) + w / 2 * np.sin(yaw),
                y - length / 2 * np.sin(yaw) - w / 2 * np.cos(yaw),
            ],
            [
                x - length / 2 * np.cos(yaw) - w / 2 * np.sin(yaw),
                y - length / 2 * np.sin(yaw) + w / 2 * np.cos(yaw),
            ],
        ]
    )

    corners = np.transpose(corners, (2, 0, 1))
    num_boxes = corners.shape[0]
    corners_flat = corners.reshape(-1, 2)

    heights = get_raster_values_at_coords(corners_flat, sim2, height_map)
    heights = heights.reshape(num_boxes, 4)

    A = np.stack([corners[..., 0], corners[..., 1], np.ones((num_boxes, 4))], axis=2)
    b = heights
    plane_coeffs = np.array(
        [np.linalg.lstsq(A[i], b[i], rcond=None)[0] for i in range(num_boxes)]
    )

    normals = np.column_stack(
        [-plane_coeffs[:, 0], -plane_coeffs[:, 1], np.ones(num_boxes)]
    )
    normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

    pitch = np.arctan2(normals[:, 0], normals[:, 2])
    roll = np.arctan2(normals[:, 1], normals[:, 2])
    eular_angles = np.stack([yaw, pitch, roll], axis=1)
    z_coords = np.mean(heights, axis=1) + h / 2

    return eular_angles.squeeze(), z_coords.squeeze()


def get_9dof_state(height_map: np.ndarray, sim2: Sim2, agent_state: np.ndarray):
    eular_angles, z_coords = calculate_box_eular_angle_and_z(
        height_map, sim2, agent_state
    )
    x, y, length, w, h, _ = agent_state.T
    yaw, pitch, roll = eular_angles.T
    bbox_9dof = np.array(
        [x, y, z_coords, length, w, h, yaw, pitch, roll],
    )
    return bbox_9dof


def get_quat_from_yaw_pitch_roll(eular_angles: np.ndarray) -> Quaternion:
    yaw, pitch, roll = eular_angles
    quat = (
        Quaternion(axis=[0, 0, 1], angle=yaw)
        * Quaternion(axis=[0, 1, 0], angle=pitch)
        * Quaternion(axis=[1, 0, 0], angle=roll)
    )
    return quat


def calculate_agent2bbox(bbox9dof: np.ndarray, agent2global: np.ndarray):
    quat = get_quat_from_yaw_pitch_roll(bbox9dof[6:9])
    bbox9dof2global_rotation = quat.rotation_matrix
    bbox9dof2global_translation = bbox9dof[:3]

    global2bbox = np.eye(4)
    global2bbox[:3, :3] = bbox9dof2global_rotation.T
    global2bbox[:3, 3] = -bbox9dof2global_rotation.T @ bbox9dof2global_translation

    agent2bbox = global2bbox @ agent2global
    return agent2bbox


def get_bbox2global(height_map: np.ndarray, sim2: Sim2, agent_state: np.ndarray):
    bbox9dof = get_9dof_state(height_map, sim2, agent_state)
    eular_angles = bbox9dof[6:9]
    quat = get_quat_from_yaw_pitch_roll(eular_angles)
    trans = bbox9dof[:3]
    box2global = np.eye(4)
    box2global[:3, :3] = quat.rotation_matrix
    box2global[:3, 3] = trans
    return box2global
