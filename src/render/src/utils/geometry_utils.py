import numpy as np
from pyquaternion import Quaternion
from scipy.optimize import minimize
import torch

from src.utils.alpasim_utils.geometry_utils import Sim2

def get_road_plane_from_traj(positions, weights):
    """
    Get the road plane from the trajectory.
    Args:
        positions: [N, 3] tensor. The trajectory positions.
        weight: the weight for the optimization.
    Returns:
        road_plane: [3] tensor, [a, b, c]. The road plane, z = ax + by + c
    """
    z = positions[:, -1]
    x = positions[:, :-1]

    def objective(params):
        ones = torch.ones(x.shape[0], 1)
        x_hat = torch.cat([x, ones], dim=1)
        z_pred = params * x_hat

        dist_cost = torch.mean(weights * (z - z_pred)**2)
        return dist_cost
    
    init_params = torch.ones(positions.shape[1])
    init_params[-1] = z.min()
    result = minimize(objective, init_params, method='Nelder-Mead')
    return result.x

def get_raster_values_at_coords(
    points_xy: np.ndarray, sim2: Sim2, np_image: np.ndarray, fill_value=np.nan
):
    # Note: we do NOT round here, because we need to enforce scaled discretization.
    city_coords = points_xy[:, :2]

    npyimage_coords = sim2.transform_from(city_coords)
    npyimage_coords = npyimage_coords.astype(np.int64)

    # out of bounds values will default to the fill value, and will not be indexed into the array.
    # index in at (x,y) locations, which are (y,x) in the image
    raster_values = np.full((npyimage_coords.shape[0]), fill_value)
    # generate boolean array indicating whether the value at each index represents a valid coordinate.
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

    # Extract 3D box parameters
    x, y, l, w, h, yaw = box.T

    l = l / 1.8  # 1.8 is the approximate ratio of the length of the car box to the wheel base

    # Create four corner points for the box in global coordinates
    corners = np.array([
        [x + l / 2 * np.cos(yaw) - w / 2 * np.sin(yaw), y + l / 2 * np.sin(yaw) + w / 2 * np.cos(yaw)],
        [x + l / 2 * np.cos(yaw) + w / 2 * np.sin(yaw), y + l / 2 * np.sin(yaw) - w / 2 * np.cos(yaw)],
        [x - l / 2 * np.cos(yaw) + w / 2 * np.sin(yaw), y - l / 2 * np.sin(yaw) - w / 2 * np.cos(yaw)],
        [x - l / 2 * np.cos(yaw) - w / 2 * np.sin(yaw), y - l / 2 * np.sin(yaw) + w / 2 * np.cos(yaw)],
    ])

    corners = np.transpose(corners, (2, 0, 1))
    num_boxes = corners.shape[0]
    corners_flat = corners.reshape(-1, 2)

    heights = get_raster_values_at_coords(corners_flat, sim2, height_map)
    heights = heights.reshape(num_boxes, 4)

    # Fit a plane to the four corner points using least squares
    # Solve for plane coefficients (a, b, c) where z = ax + by + c
    A = np.stack([corners[..., 0],
                  corners[..., 1],
                  np.ones((num_boxes, 4))],
                  axis=2)
    b = heights
    plane_coeffs = np.array([np.linalg.lstsq(A[i], b[i], rcond=None)[0] for i in range(num_boxes)])

    # Calculate pitch and roll from plane normal vector
    # Normal vector is [-a, -b, 1] for plane ax + by - z + c = 0
    normals = np.column_stack([-plane_coeffs[:, 0], -plane_coeffs[:, 1], np.ones(num_boxes)])
    normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

    # Pitch is angle between normal projection on xz-plane and z-axis
    # Roll is angle between normal projection on yz-plane and z-axis
    pitch = np.arctan2(normals[:, 0], normals[:, 2])
    roll = np.arctan2(normals[:, 1], normals[:, 2])
    eular_angles = np.stack([yaw, pitch, roll], axis=1)
    z_coords = np.mean(heights, axis=1) + h / 2

    return eular_angles.squeeze(), z_coords.squeeze()

def get_9dof_state(height_map: np.ndarray, sim2: Sim2, agent_state: np.ndarray):
    eular_angles, z_coords = calculate_box_eular_angle_and_z(height_map, sim2, agent_state)
    x, y, l, w, h, _ = agent_state.T
    yaw, pitch, roll = eular_angles.T
    bbox_9dof = np.array(
        [x, y, z_coords, l, w, h, yaw, pitch, roll],
    )
    return bbox_9dof

def calculate_agent2bbox(bbox9dof: np.ndarray, agent2global: np.ndarray):
    """
    """
    quat = get_quat_from_yaw_pitch_roll(bbox9dof[6:9])
    bbox9dof2global_rotation = quat.rotation_matrix
    bbox9dof2global_translation = bbox9dof[:3]

    global2bbox = np.eye(4)
    global2bbox[:3, :3] = bbox9dof2global_rotation.T
    global2bbox[:3, 3] = -bbox9dof2global_rotation.T @ bbox9dof2global_translation

    agent2bbox = global2bbox @ agent2global
    return agent2bbox

def get_quat_from_yaw_pitch_roll(eular_angles: np.ndarray) -> Quaternion:
    yaw, pitch, roll = eular_angles
    quat = Quaternion(axis=[0, 0, 1], angle=yaw) * \
           Quaternion(axis=[0, 1, 0], angle=pitch) * \
           Quaternion(axis=[1, 0, 0], angle=roll)
    return quat

def get_bbox2global(height_map: np.ndarray, sim2: Sim2, agent_state: np.ndarray):
    bbox9dof = get_9dof_state(height_map, sim2, agent_state)
    eular_angles = bbox9dof[6:9]
    quat = get_quat_from_yaw_pitch_roll(eular_angles)
    trans = bbox9dof[:3]
    box2global = np.eye(4)
    box2global[:3, :3] = quat.rotation_matrix
    box2global[:3, 3] = trans
    return box2global
