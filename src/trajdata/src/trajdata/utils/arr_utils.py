# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from enum import IntEnum
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class PadDirection(IntEnum):
    BEFORE = 0
    AFTER = 1


def convert_with_dir(
    seq: np.ndarray, dtype: torch.dtype, time_dim: int, pad_dir: PadDirection
) -> Tensor:
    if pad_dir == PadDirection.BEFORE:
        return torch.as_tensor(seq, dtype=dtype).flip(time_dim)

    return torch.as_tensor(seq, dtype=dtype)


def pad_with_dir(
    seq_list: List[Tensor], time_dim: int, pad_dir: PadDirection, **kwargs
) -> Tensor:
    if pad_dir == PadDirection.BEFORE:
        return pad_sequence(
            seq_list,
            **kwargs,
        ).flip(time_dim)

    return pad_sequence(
        seq_list,
        **kwargs,
    )


def pad_sequences(
    seq_list: List[np.ndarray],
    dtype: torch.dtype,
    time_dim: int,
    pad_dir: PadDirection,
    **kwargs,
) -> Tensor:
    return pad_with_dir(
        [convert_with_dir(seq, dtype, time_dim, pad_dir) for seq in seq_list],
        time_dim,
        pad_dir,
        **kwargs,
    )


def mask_up_to(lens: Tensor, delta: int = 0, max_len: Optional[int] = None) -> Tensor:
    """Exclusive.

    Args:
        lens (Tensor): _description_
        delta (int, optional): _description_. Defaults to 0.

    Returns:
        Tensor: _description_
    """
    if max_len is None:
        max_len = lens.max()

    arange_t: Tensor = torch.arange(
        max_len, dtype=lens.dtype, device=lens.device
    ).expand(*lens.shape, -1)

    return arange_t < (lens.unsqueeze(-1) + delta)


def vrange(starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Create concatenated ranges of integers for multiple start/stop

    Parameters:
        starts (1-D array): starts for each range
        stops (1-D array): stops for each range (same shape as starts)

    Returns:
        numpy.ndarray: concatenated ranges

    For example:

        >>> starts = np.array([1, 3, 4, 6])
        >>> stops  = np.array([1, 5, 7, 6])
        >>> vrange(starts, stops)
        array([3, 4, 4, 5, 6])

    """
    lens = stops - starts
    return np.repeat(stops - lens.cumsum(), lens) + np.arange(lens.sum())


def angle_wrap(
    radians: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """This function wraps angles to lie within [-pi, pi).

    Args:
        radians (np.ndarray): The input array of angles (in radians).

    Returns:
        np.ndarray: Wrapped angles that lie within [-pi, pi).
    """
    return (radians + np.pi) % (2 * np.pi) - np.pi


def rotation_matrix(angle: Union[float, np.ndarray]) -> np.ndarray:
    """Creates one or many 2D rotation matrices.

    Args:
        angle (Union[float, np.ndarray]): The angle to rotate points by.
            if float, returns 2x2 matrix
            if np.ndarray, expects shape [...], and returns [...,2,2] array

    Returns:
        np.ndarray: The 2x2 rotation matri(x/ces).
    """
    batch_dims = 0
    if isinstance(angle, np.ndarray):
        batch_dims = angle.ndim
        angle = angle

    rotmat: np.ndarray = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    return rotmat.transpose(*np.arange(2, batch_dims + 2), 0, 1)


def transform_matrices(angles: Tensor, translations: Optional[Tensor]) -> Tensor:
    """Creates a 3x3 transformation matrix for each angle and translation in the input.

    Args:
        angles (Tensor): The (...)-shaped angles tensor to rotate points by (in radians).
        translations (Tensor): The (...,2)-shaped translations to shift points by.

    Returns:
        Tensor: The Nx3x3 transformation matrices.
    """
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)
    last_rows = (
        torch.tensor([0.0, 0.0, 1.0], dtype=angles.dtype, device=angles.device)
        .view([1] * angles.ndim + [3])
        .expand(list(angles.shape) + [-1])
    )

    if translations is None:
        trans_x = torch.zeros_like(angles)
        trans_y = trans_x
    else:
        trans_x, trans_y = torch.unbind(translations, dim=-1)

    return torch.stack(
        [
            torch.stack([cos_vals, -sin_vals, trans_x], dim=-1),
            torch.stack([sin_vals, cos_vals, trans_y], dim=-1),
            last_rows,
        ],
        dim=-2,
    )


def transform_coords_2d_np(
    coords: np.ndarray,
    offset: Optional[np.ndarray] = None,
    angle: Optional[np.ndarray] = None,
    rot_mat: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Args:
        coords (np.ndarray): [..., 2] coordinates
        offset (Optional[np.ndarray], optional): [..., 2] offset to translate. Defaults to None.
        angle (Optional[np.ndarray], optional): [...] angle to rotate by. Defaults to None.
        rot_mat (Optional[np.ndarray], optional): [..., 2,2] rotation matrix to apply. Defaults to None.
            If rot_mat is given, angle is ignored.

    Returns:
        np.ndarray: transformed coords
    """
    if rot_mat is None and angle is not None:
        rot_mat = rotation_matrix(angle)

    if rot_mat is not None:
        coords = np.einsum("...ij,...j->...i", rot_mat, coords)

    if offset is not None:
        coords += offset

    return coords


def transform_coords_np(
    coords: np.ndarray, tf_mat: np.ndarray, translate: bool = True
) -> np.ndarray:
    """
    Returns coords after transforming them according to the transformation matrix tf_mat

    Args:
        coords (np.ndarray): batch of points [..., d]
        tf_mat (np.ndarray): nd affine transformation matrix [..., d+1, d+1]
            or [d+1, d+1] if the same transformation should be applied to all points

    Returns:
        np.ndarray: transformed points [..., d]
    """
    if coords.ndim == (tf_mat.ndim - 1):
        transformed = np.einsum("...jk,...k->...j", tf_mat[..., :-1, :-1], coords)
        if translate:
            transformed += tf_mat[..., :-1, -1]
    elif tf_mat.ndim == 2:
        transformed = np.einsum("jk,...k->...j", tf_mat[:-1, :-1], coords)
        if translate:
            transformed += tf_mat[None, :-1, -1]
    else:
        raise ValueError("Batch dims of tf_mat must match coords")

    return transformed


def transform_angles_np(angles: np.ndarray, tf_mat: np.ndarray) -> np.ndarray:
    """
    Returns angles after transforming them according to the transformation matrix tf_mat

    Args:
        angles (np.ndarray): batch of angles [...]
        tf_mat (np.ndarray): nd affine transformation matrix [..., d+1, d+1]
            or [d+1, d+1] if the same transformation should be applied to all points

    Returns:
        np.ndarray: transformed angles [...]
    """
    cos_vals, sin_vals = tf_mat[..., 0, 0], tf_mat[..., 1, 0]
    rot_angle = np.arctan2(sin_vals, cos_vals)
    transformed_angles = angles + rot_angle
    transformed_angles = angle_wrap(transformed_angles)
    return transformed_angles


def transform_xyh_np(xyh: np.ndarray, tf_mat: np.ndarray) -> np.ndarray:
    """
    Returns transformed set of xyh points

    Args:
        xyh (np.ndarray): shape [...,3]
        tf_mat (np.ndarray): shape [...,3,3]
    """
    transformed_xy = transform_coords_np(xyh[..., :-1], tf_mat)
    transformed_angles = transform_angles_np(xyh[..., -1], tf_mat)
    return np.concatenate([transformed_xy, transformed_angles[..., None]], axis=-1)


def transform_xyh_torch(xyh: torch.Tensor, tf_mat: torch.Tensor) -> torch.Tensor:
    """
    Returns transformed set of xyh points

    Args:
        xyh (torch.Tensor): shape [...,3]
        tf_mat (torch.Tensor): shape [...,3,3]
    """
    transformed_xy = batch_nd_transform_points_pt(xyh[..., :2], tf_mat)
    transformed_angles = batch_nd_transform_angles_pt(xyh[..., 2], tf_mat)
    return torch.cat([transformed_xy, transformed_angles[..., None]], dim=-1)


# -------- TODO redundant transforms, remove them


def batch_nd_transform_points_np(points: np.ndarray, Mat: np.ndarray) -> np.ndarray:
    ndim = Mat.shape[-1] - 1
    batch = list(range(Mat.ndim - 2)) + [Mat.ndim - 1] + [Mat.ndim - 2]
    Mat = np.transpose(Mat, batch)
    if points.ndim == Mat.ndim - 1:
        return (points[..., np.newaxis, :] @ Mat[..., :ndim, :ndim]).squeeze(-2) + Mat[
            ..., -1:, :ndim
        ].squeeze(-2)
    elif points.ndim == Mat.ndim:
        return (
            (points[..., np.newaxis, :] @ Mat[..., np.newaxis, :ndim, :ndim])
            + Mat[..., np.newaxis, -1:, :ndim]
        ).squeeze(-2)
    else:
        raise Exception("wrong shape")


def batch_nd_transform_points_pt(
    points: torch.Tensor, Mat: torch.Tensor
) -> torch.Tensor:
    ndim = Mat.shape[-1] - 1
    Mat = torch.transpose(Mat, -1, -2)
    if points.ndim == Mat.ndim - 1:
        return (points[..., np.newaxis, :] @ Mat[..., :ndim, :ndim]).squeeze(-2) + Mat[
            ..., -1:, :ndim
        ].squeeze(-2)
    elif points.ndim == Mat.ndim:
        return (
            (points[..., np.newaxis, :] @ Mat[..., np.newaxis, :ndim, :ndim])
            + Mat[..., np.newaxis, -1:, :ndim]
        ).squeeze(-2)
    elif points.ndim == Mat.ndim + 1:
        return (
            (
                points[..., np.newaxis, :]
                @ Mat[..., np.newaxis, np.newaxis, :ndim, :ndim]
            )
            + Mat[..., np.newaxis, np.newaxis, -1:, :ndim]
        ).squeeze(-2)
    else:
        raise Exception("wrong shape")


def batch_nd_transform_angles_np(angles: np.ndarray, Mat: np.ndarray) -> np.ndarray:
    cos_vals, sin_vals = Mat[..., 0, 0], Mat[..., 1, 0]
    rot_angle = np.arctan2(sin_vals, cos_vals)
    angles = angles + rot_angle
    angles = angle_wrap(angles)
    return angles


def batch_nd_transform_angles_pt(
    angles: torch.Tensor, Mat: torch.Tensor
) -> torch.Tensor:
    cos_vals, sin_vals = Mat[..., 0, 0], Mat[..., 1, 0]
    rot_angle = torch.arctan2(sin_vals, cos_vals)
    if rot_angle.ndim > angles.ndim:
        raise ValueError("wrong shape")
    while rot_angle.ndim < angles.ndim:
        rot_angle = rot_angle.unsqueeze(-1)
    angles = angles + rot_angle
    angles = angle_wrap(angles)
    return angles


def batch_nd_transform_points_angles_np(
    points_angles: np.ndarray, Mat: np.ndarray
) -> np.ndarray:
    assert points_angles.shape[-1] == 3
    points = batch_nd_transform_points_np(points_angles[..., :2], Mat)
    angles = batch_nd_transform_angles_np(points_angles[..., 2:3], Mat)
    points_angles = np.concatenate([points, angles], axis=-1)
    return points_angles


def batch_nd_transform_points_angles_pt(
    points_angles: torch.Tensor, Mat: torch.Tensor
) -> torch.Tensor:
    assert points_angles.shape[-1] == 3
    points = batch_nd_transform_points_pt(points_angles[..., :2], Mat)
    angles = batch_nd_transform_angles_pt(points_angles[..., 2:3], Mat)
    points_angles = torch.concat([points, angles], axis=-1)
    return points_angles


def batch_nd_transform_xyvvaahh_pt(
    traj_xyvvaahh: torch.Tensor, tf: torch.Tensor
) -> torch.Tensor:
    """
    traj_xyvvaahh: [..., state_dim] where state_dim = [x, y, vx, vy, ax, ay, sinh, cosh]
    This is the state representation used in AgentBatch and SceneBatch.
    """
    rot_only_tf = tf.clone()
    rot_only_tf[..., :2, -1] = 0.0

    xy, vv, aa, hh = torch.split(traj_xyvvaahh, (2, 2, 2, 2), dim=-1)
    xy = batch_nd_transform_points_pt(xy, tf)
    vv = batch_nd_transform_points_pt(vv, rot_only_tf)
    aa = batch_nd_transform_points_pt(aa, rot_only_tf)
    # hh: sinh, cosh instead of cosh, sinh, so we use flip
    hh = batch_nd_transform_points_pt(hh.flip(-1), rot_only_tf).flip(-1)

    return torch.concat((xy, vv, aa, hh), dim=-1)


# -------- end of redundant transforms


def transform_xyh_torch(xyh: torch.Tensor, tf_mat: torch.Tensor) -> torch.Tensor:
    """
    Returns transformed set of xyh points

    Args:
        xyh (torch.Tensor): shape [...,3]
        tf_mat (torch.Tensor): shape [...,3,3]
    """
    transformed_xy = batch_nd_transform_points_pt(xyh[..., :2], tf_mat)
    transformed_angles = batch_nd_transform_angles_pt(xyh[..., 2], tf_mat)
    return torch.cat([transformed_xy, transformed_angles[..., None]], dim=-1)


# -------- TODO redundant transforms, remove them


def batch_nd_transform_points_np(points: np.ndarray, Mat: np.ndarray) -> np.ndarray:
    ndim = Mat.shape[-1] - 1
    batch = list(range(Mat.ndim - 2)) + [Mat.ndim - 1] + [Mat.ndim - 2]
    Mat = np.transpose(Mat, batch)
    if points.ndim == Mat.ndim - 1:
        return (points[..., np.newaxis, :] @ Mat[..., :ndim, :ndim]).squeeze(-2) + Mat[
            ..., -1:, :ndim
        ].squeeze(-2)
    elif points.ndim == Mat.ndim:
        return (
            (points[..., np.newaxis, :] @ Mat[..., np.newaxis, :ndim, :ndim])
            + Mat[..., np.newaxis, -1:, :ndim]
        ).squeeze(-2)
    else:
        raise Exception("wrong shape")


def batch_nd_transform_points_pt(
    points: torch.Tensor, Mat: torch.Tensor
) -> torch.Tensor:
    ndim = Mat.shape[-1] - 1
    Mat = torch.transpose(Mat, -1, -2)
    if points.ndim == Mat.ndim - 1:
        return (points[..., np.newaxis, :] @ Mat[..., :ndim, :ndim]).squeeze(-2) + Mat[
            ..., -1:, :ndim
        ].squeeze(-2)
    elif points.ndim == Mat.ndim:
        return (
            (points[..., np.newaxis, :] @ Mat[..., np.newaxis, :ndim, :ndim])
            + Mat[..., np.newaxis, -1:, :ndim]
        ).squeeze(-2)
    elif points.ndim == Mat.ndim + 1:
        return (
            (
                points[..., np.newaxis, :]
                @ Mat[..., np.newaxis, np.newaxis, :ndim, :ndim]
            )
            + Mat[..., np.newaxis, np.newaxis, -1:, :ndim]
        ).squeeze(-2)
    else:
        raise Exception("wrong shape")


def batch_nd_transform_angles_np(angles: np.ndarray, Mat: np.ndarray) -> np.ndarray:
    cos_vals, sin_vals = Mat[..., 0, 0], Mat[..., 1, 0]
    rot_angle = np.arctan2(sin_vals, cos_vals)
    angles = angles + rot_angle
    angles = angle_wrap(angles)
    return angles


def batch_nd_transform_angles_pt(
    angles: torch.Tensor, Mat: torch.Tensor
) -> torch.Tensor:
    cos_vals, sin_vals = Mat[..., 0, 0], Mat[..., 1, 0]
    rot_angle = torch.arctan2(sin_vals, cos_vals)
    if rot_angle.ndim > angles.ndim:
        raise ValueError("wrong shape")
    while rot_angle.ndim < angles.ndim:
        rot_angle = rot_angle.unsqueeze(-1)
    angles = angles + rot_angle
    angles = angle_wrap(angles)
    return angles


def batch_nd_transform_points_angles_np(
    points_angles: np.ndarray, Mat: np.ndarray
) -> np.ndarray:
    assert points_angles.shape[-1] == 3
    points = batch_nd_transform_points_np(points_angles[..., :2], Mat)
    angles = batch_nd_transform_angles_np(points_angles[..., 2:3], Mat)
    points_angles = np.concatenate([points, angles], axis=-1)
    return points_angles


def batch_nd_transform_points_angles_pt(
    points_angles: torch.Tensor, Mat: torch.Tensor
) -> torch.Tensor:
    assert points_angles.shape[-1] == 3
    points = batch_nd_transform_points_pt(points_angles[..., :2], Mat)
    angles = batch_nd_transform_angles_pt(points_angles[..., 2:3], Mat)
    points_angles = torch.concat([points, angles], axis=-1)
    return points_angles


def batch_nd_transform_xyvvaahh_pt(
    traj_xyvvaahh: torch.Tensor, tf: torch.Tensor
) -> torch.Tensor:
    """
    traj_xyvvaahh: [..., state_dim] where state_dim = [x, y, vx, vy, ax, ay, sinh, cosh]
    This is the state representation used in AgentBatch and SceneBatch.
    """
    rot_only_tf = tf.clone()
    rot_only_tf[..., :2, -1] = 0.0

    xy, vv, aa, hh = torch.split(traj_xyvvaahh, (2, 2, 2, 2), dim=-1)
    xy = batch_nd_transform_points_pt(xy, tf)
    vv = batch_nd_transform_points_pt(vv, rot_only_tf)
    aa = batch_nd_transform_points_pt(aa, rot_only_tf)
    # hh: sinh, cosh instead of cosh, sinh, so we use flip
    hh = batch_nd_transform_points_pt(hh.flip(-1), rot_only_tf).flip(-1)

    return torch.concat((xy, vv, aa, hh), dim=-1)


# -------- end of redundant transforms


def agent_aware_diff(values: np.ndarray, agent_ids: np.ndarray) -> np.ndarray:
    values_diff: np.ndarray = np.diff(
        values, axis=0, prepend=values[[0]] - (values[[1]] - values[[0]])
    )

    # The point of the border mask is to catch data like this:
    # index    agent_id     vx    vy
    #     0           1    7.3   9.1
    #     1           2    0.0   0.0
    #                  ...
    # As implemented, we're not touching the very last row (we don't care anyways
    # for agents detected only once at a single timestep) and we would currently only
    # return index 0 (since we chop off the top with the 1: in the slice below), but
    # we want to return 1 so that's why the + 1 at the end.
    border_mask: np.ndarray = np.nonzero(agent_ids[1:-1] != agent_ids[:-2])[0] + 1
    values_diff[border_mask] = values_diff[border_mask + 1]

    return values_diff


def batch_proj(x, line):
    # x:[batch,3], line:[batch,N,3]
    line_length = line.shape[-2]
    batch_dim = x.ndim - 1
    if isinstance(x, torch.Tensor):
        delta = line[..., 0:2] - torch.unsqueeze(x[..., 0:2], dim=-2).repeat(
            *([1] * batch_dim), line_length, 1
        )
        dis = torch.linalg.norm(delta, axis=-1)
        idx0 = torch.argmin(dis, dim=-1)
        idx = idx0.view(*line.shape[:-2], 1, 1).repeat(
            *([1] * (batch_dim + 1)), line.shape[-1]
        )
        line_min = torch.squeeze(torch.gather(line, -2, idx), dim=-2)
        dx = x[..., None, 0] - line[..., 0]
        dy = x[..., None, 1] - line[..., 1]
        delta_y = -dx * torch.sin(line_min[..., None, 2]) + dy * torch.cos(
            line_min[..., None, 2]
        )
        delta_x = dx * torch.cos(line_min[..., None, 2]) + dy * torch.sin(
            line_min[..., None, 2]
        )

        delta_psi = angle_wrap(x[..., 2] - line_min[..., 2])

        return (
            delta_x,
            delta_y,
            torch.unsqueeze(delta_psi, dim=-1),
        )

    elif isinstance(x, np.ndarray):
        delta = line[..., 0:2] - np.repeat(
            x[..., np.newaxis, 0:2], line_length, axis=-2
        )
        dis = np.linalg.norm(delta, axis=-1)
        idx0 = np.argmin(dis, axis=-1)
        idx = idx0.reshape(*line.shape[:-2], 1, 1).repeat(line.shape[-1], axis=-1)
        line_min = np.squeeze(np.take_along_axis(line, idx, axis=-2), axis=-2)
        dx = x[..., None, 0] - line[..., 0]
        dy = x[..., None, 1] - line[..., 1]
        delta_y = -dx * np.sin(line_min[..., None, 2]) + dy * np.cos(
            line_min[..., None, 2]
        )
        delta_x = dx * np.cos(line_min[..., None, 2]) + dy * np.sin(
            line_min[..., None, 2]
        )
        delta_psi = angle_wrap(x[..., 2] - line_min[..., 2])
        return (
            delta_x,
            delta_y,
            np.expand_dims(delta_psi, axis=-1),
        )


def quaternion_to_yaw(q: np.ndarray):
    # From https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L1025
    return np.arctan2(
        2 * (q[..., 0] * q[..., 3] - q[..., 1] * q[..., 2]),
        1 - 2 * (q[..., 2] ** 2 + q[..., 3] ** 2),
    )


def batch_select(x: torch.Tensor, index: torch.Tensor, batch_dims: int) -> torch.Tensor:
    # Indexing into tensor, treating the first `batch_dims` dimensions as batch.
    # Kind of: output[..., k] = x[..., index[...]]

    assert index.ndim >= batch_dims
    assert index.ndim <= x.ndim
    assert x.shape[:batch_dims] == index.shape[:batch_dims]

    batch_shape = x.shape[:batch_dims]
    x_flat = x.reshape(-1, *x.shape[batch_dims:])
    index_flat = index.reshape(-1, *index.shape[batch_dims:])
    x_flat = x_flat[torch.arange(x_flat.shape[0]), index_flat]
    x = x_flat.reshape(*batch_shape, *x_flat.shape[1:])

    return x


def roll_with_tensor(mat: torch.Tensor, shifts: torch.LongTensor, dim: int):
    if dim < 0:
        dim = mat.ndim + dim
    arange1 = torch.arange(mat.shape[dim], device=shifts.device)
    expanded_shape = [1] * dim + [-1] + [1] * (mat.ndim - dim - 1)
    arange1 = arange1.view(expanded_shape).expand(mat.shape)
    if shifts.ndim == 1:
        shifts = shifts.view([1] * (dim - 1) + [-1])
    # TODO assert that shift dimenesions either match mat or 1
    shifts = shifts.view(list(shifts.shape) + [1] * (mat.ndim - dim))

    arange2 = (arange1 - shifts) % mat.shape[dim]
    # print(arange2)
    return torch.gather(mat, dim, arange2)


def round_2pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def batch_proj(x, line):
    # x:[batch,3], line:[batch,N,3]
    line_length = line.shape[-2]
    batch_dim = x.ndim - 1
    if isinstance(x, torch.Tensor):
        delta = line[..., 0:2] - torch.unsqueeze(x[..., 0:2], dim=-2).repeat(
            *([1] * batch_dim), line_length, 1
        )
        dis = torch.linalg.norm(delta, axis=-1)
        idx0 = torch.argmin(dis, dim=-1)
        idx = idx0.view(*line.shape[:-2], 1, 1).repeat(
            *([1] * (batch_dim + 1)), line.shape[-1]
        )
        line_min = torch.squeeze(torch.gather(line, -2, idx), dim=-2)
        dx = x[..., None, 0] - line[..., 0]
        dy = x[..., None, 1] - line[..., 1]
        delta_y = -dx * torch.sin(line_min[..., None, 2]) + dy * torch.cos(
            line_min[..., None, 2]
        )
        delta_x = dx * torch.cos(line_min[..., None, 2]) + dy * torch.sin(
            line_min[..., None, 2]
        )

        delta_psi = round_2pi(x[..., 2] - line_min[..., 2])

        return (
            delta_x,
            delta_y,
            torch.unsqueeze(delta_psi, dim=-1),
        )
    elif isinstance(x, np.ndarray):
        delta = line[..., 0:2] - np.repeat(
            x[..., np.newaxis, 0:2], line_length, axis=-2
        )
        dis = np.linalg.norm(delta, axis=-1)
        idx0 = np.argmin(dis, axis=-1)
        idx = idx0.reshape(*line.shape[:-2], 1, 1).repeat(line.shape[-1], axis=-1)
        line_min = np.squeeze(np.take_along_axis(line, idx, axis=-2), axis=-2)
        dx = x[..., None, 0] - line[..., 0]
        dy = x[..., None, 1] - line[..., 1]
        delta_y = -dx * np.sin(line_min[..., None, 2]) + dy * np.cos(
            line_min[..., None, 2]
        )
        delta_x = dx * np.cos(line_min[..., None, 2]) + dy * np.sin(
            line_min[..., None, 2]
        )

        delta_psi = round_2pi(x[..., 2] - line_min[..., 2])
        return (
            delta_x,
            delta_y,
            np.expand_dims(delta_psi, axis=-1),
        )


def get_close_lanes(radius, ego_xyh, vec_map, num_pts):
    # obtain close lanes, their distance to the ego
    close_lanes = []
    while len(close_lanes) == 0:
        close_lanes = vec_map.get_lanes_within(ego_xyh, radius)
        radius += 20
    dis = list()
    lane_pts = np.stack(
        [lane.center.interpolate(num_pts).points[:, [0, 1, 3]] for lane in close_lanes],
        0,
    )
    dx, dy, dh = batch_proj(ego_xyh[None].repeat(lane_pts.shape[0], 0), lane_pts)

    idx = np.abs(dx).argmin(axis=1)
    # hausdorff distance to the lane (longitudinal)
    x_dis = np.take_along_axis(np.abs(dx), idx[:, None], axis=1).squeeze(1)
    x_dis[(dx.min(1) < 0) & (dx.max(1) > 0)] = 0

    y_dis = np.take_along_axis(np.abs(dy), idx[:, None], axis=1).squeeze(1)

    # distance metric to the lane (combining x,y)
    dis = x_dis + y_dis

    return close_lanes, dis


def get_close_road_edges(radius, ego_xyh, vec_map, num_pts):
    # obtain close lanes, their distance to the ego
    close_road_edges = []
    while len(close_road_edges) == 0:
        close_road_edges = vec_map.get_road_edges_within(ego_xyh, radius)
        radius += 20
    dis = list()
    road_edge_pts = np.stack(
        [
            road_edge.polyline.interpolate(num_pts).points[:, [0, 1, 3]]
            for road_edge in close_road_edges
        ],
        0,
    )
    dx, dy, dh = batch_proj(
        ego_xyh[None].repeat(road_edge_pts.shape[0], 0), road_edge_pts
    )

    idx = np.abs(dx).argmin(axis=1)
    # hausdorff distance to the lane (longitudinal)
    x_dis = np.take_along_axis(np.abs(dx), idx[:, None], axis=1).squeeze(1)
    x_dis[(dx.min(1) < 0) & (dx.max(1) > 0)] = 0

    y_dis = np.take_along_axis(np.abs(dy), idx[:, None], axis=1).squeeze(1)

    # distance metric to the lane (combining x,y)
    dis = x_dis + y_dis

    return close_road_edges, dis
