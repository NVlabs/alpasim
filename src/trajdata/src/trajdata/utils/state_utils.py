# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from typing import Optional

import numpy as np

from trajdata.data_structures.state import StateArray, StateTensor
from trajdata.utils.arr_utils import (
    angle_wrap,
    rotation_matrix,
    transform_angles_np,
    transform_coords_2d_np,
    transform_coords_np,
)


def transform_state_np_2d(state: StateArray, tf_mat_2d: np.ndarray):
    """
    Transforms a state into another coordinate frame
    assumes center has dim 2 (xy shift) or shape 6 normalizes derivatives as well
    """
    new_state = state.copy()
    attributes = state._format_dict.keys()
    if "x" in attributes and "y" in attributes:
        # transform xy position with translation and rotation
        new_state.position = transform_coords_np(state.position, tf_mat_2d)
    if "xd" in attributes and "yd" in attributes:
        # transform velocities
        new_state.velocity = transform_coords_np(
            state.velocity, tf_mat_2d, translate=False
        )
    if "xdd" in attributes and "ydd" in attributes:
        # transform acceleration
        new_state.acceleration = transform_coords_np(
            state.acceleration, tf_mat_2d, translate=False
        )
    if "c" in attributes and "s" in attributes:
        new_state.heading_vector = transform_coords_np(
            state.heading_vector, tf_mat_2d, translate=False
        )
    if "h" in attributes:
        new_state.heading = transform_angles_np(state.heading, tf_mat_2d)

    return new_state


def convert_to_frame_state(
    state: StateArray,
    stationary: bool = True,
    grounded: bool = True,
) -> StateArray:
    """
    Returns a StateArray corresponding to a frame centered around the passed in State
    """
    frame: StateArray = state.copy()
    attributes = state._format_dict.keys()
    if stationary:
        if "xd" in attributes and "yd" in attributes:
            frame.velocity = 0
        if "xdd" in attributes and "ydd" in attributes:
            frame.acceleration = 0
    if grounded:
        if "z" in attributes:
            frame.set_attr("z", 0)

    return frame


def transform_to_frame(
    state: StateArray, frame_state: StateArray, rot_mat: Optional[np.ndarray] = None
) -> StateArray:
    """
    Returns state with coordinates relative to a frame with state frame_state.
    Does not modify state in place.

    Args:
        state (StateArray): state to transform in world coordinates
        frame_state (StateArray): state of frame in world coordinates
        rot_mat Optional[nd.array]: rotation matrix A such that c = A @ b returns coordinates in the new frame
            if not given, it is computed frome frame_state
    """
    new_state = state.copy()
    attributes = state._format_dict.keys()

    frame_heading = frame_state.heading[..., 0]
    if rot_mat is None:
        rot_mat = rotation_matrix(-frame_heading)

    if "x" in attributes and "y" in attributes:
        # transform xy position with translation and rotation
        new_state.position = transform_coords_2d_np(
            state.position, offset=-frame_state.position, rot_mat=rot_mat
        )
    if "xd" in attributes and "yd" in attributes:
        # transform velocities
        new_state.velocity = transform_coords_2d_np(
            state.velocity, offset=-frame_state.velocity, rot_mat=rot_mat
        )
    if "xdd" in attributes and "ydd" in attributes:
        # transform acceleration
        new_state.acceleration = transform_coords_2d_np(
            state.acceleration, offset=-frame_state.acceleration, rot_mat=rot_mat
        )
    if "c" in attributes and "s" in attributes:
        new_state.heading_vector = transform_coords_2d_np(
            state.heading_vector, rot_mat=rot_mat
        )
    if "h" in attributes:
        new_state.heading = angle_wrap(state.heading - frame_heading)

    return new_state


def transform_from_frame(
    state: StateArray, frame_state: StateArray, rot_mat: Optional[np.ndarray] = None
) -> StateArray:
    """
    Returns state with coordinates in world frame
    Does not modify state in place.

    Args:
        state (StateArray): state to transform in world coordinates
        frame_state (StateArray): state of frame in world coordinates
        rot_mat Optional[nd.array]: rotation matrix A such that c = A @ b returns coordinates in the new frame
            if not given, it is computed frome frame_state
    """
    new_state = state.copy()
    attributes = state._format_dict.keys()

    frame_heading = frame_state.heading[..., 0]
    if rot_mat is None:
        rot_mat = rotation_matrix(frame_heading)

    if "x" in attributes and "y" in attributes:
        # transform xy position with translation and rotation
        new_state.position = (
            transform_coords_2d_np(state.position, rot_mat=rot_mat)
            + frame_state.position
        )
    if "xd" in attributes and "yd" in attributes:
        # transform velocities
        new_state.velocity = (
            transform_coords_2d_np(
                state.velocity,
                angle=frame_heading,
            )
            + frame_state.velocity
        )
    if "xdd" in attributes and "ydd" in attributes:
        # transform acceleration
        new_state.acceleration = (
            transform_coords_2d_np(
                state.acceleration,
                angle=frame_heading,
            )
            + frame_state.acceleration
        )
    if "c" in attributes and "s" in attributes:
        new_state.heading_vector = transform_coords_2d_np(
            state.heading_vector,
            angle=frame_heading,
        )
    if "h" in attributes:
        new_state.heading = angle_wrap(state.heading + frame_heading)

    return new_state
