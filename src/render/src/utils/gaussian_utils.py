import math
import torch

@torch.jit.script
def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    assert quat.shape[-1] == 4, f"Expected quaternion shape [..., 4], got {quat.shape}"
    w, x, y, z = torch.unbind(quat, dim=-1)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    m00 = 1.0 - 2.0 * (yy + zz)
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)
    m10 = 2.0 * (xy + wz)
    m11 = 1.0 - 2.0 * (xx + zz)
    m12 = 2.0 * (yz - wx)
    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = 1.0 - 2.0 * (xx + yy)

    mat = torch.stack([m00, m01, m02, m10, m11, m12, m20, m21, m22], dim=-1)
    return mat.reshape(quat.shape[:-1] + (3, 3))

@torch.jit.script
def quat_mult(q1, q2):
    # NOTE:
    # Q1 is the quaternion that rotates the vector from the original position to the final position
    # Q2 is the quaternion that been rotated
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    w2, x2, y2, z2 = torch.unbind(q2, dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)

def interpolate_quats(q1, q2, fraction=0.5):
    """
    Interpolate between two quaternions using spherical linear interpolation (slerp).
    """
    if q1.dim() == 1:
        q1 = q1[None, ...]
    if q2.dim() == 1:
        q2 = q2[None, ...]

    q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)
    q2 = q2 / torch.norm(q2, dim=-1, keepdim=True)

    dot = (q1 * q2).sum(dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1, 1)

    q2 = torch.where(dot < 0, -q2, q2)
    dot = torch.abs(dot)

    similar_mask = dot > 0.9995
    q_interp_similar = q1 + fraction * (q2 - q1)

    theta_0 = torch.acos(dot)
    theta = theta_0 * fraction
    sin_theta = torch.sin(theta)
    sin_theta_0 = torch.sin(theta_0)
    s1 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0
    q_interp = (s1[..., None] * q1) + (s2[..., None] * q2)

    final_q_interp = torch.zeros_like(q1)
    final_q_interp = torch.where(similar_mask, q_interp_similar, q_interp)
    final_q_interp = final_q_interp / torch.norm(final_q_interp, dim=-1, keepdim=True)
    return final_q_interp

def IDFT(time, dim, input_normalized=True):
    """
    Computes the inverse discrete Fourier transform of a given time signal.
    """
    if isinstance(time, float):
        time = torch.tensor(time)
    t = time.view(-1, 1)
    idft = torch.zeros(t.shape[0], dim, dtype=t.dtype, device=t.device)
    indices = torch.arange(dim, dtype=torch.int, device=t.device)
    even_indices = indices[::2]
    odd_indices = indices[1::2]
    if input_normalized:
        idft[:, even_indices] = torch.cos(t * even_indices * 2 * math.pi / dim)
        idft[:, odd_indices] = torch.sin(t * (odd_indices + 1) * 2 * math.pi / dim)
    else:
        idft[:, even_indices] = torch.cos(t * even_indices)
        idft[:, odd_indices] = torch.sin(t * (odd_indices + 1))
    return idft

def quat_to_angle(quat, focus="yaw"):
    # quat: [N, 4]
    focus = [focus] if isinstance(focus, str) else focus
    w, x, y, z = torch.unbind(quat, dim=-1)
    return_dict = dict()
    if "yaw" in focus:
        return_dict["yaw"] = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    if "pitch" in focus:
        return_dict["pitch"] = torch.asin(2 * (w * y - z * x))
    if "roll" in focus:
        return_dict["roll"] = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    return return_dict

def angle_to_quat(yaw, pitch, roll):
    # yaw, pitch, roll: [N]
    cy = torch.cos(yaw / 2)
    sy = torch.sin(yaw / 2)
    cp = torch.cos(pitch / 2)
    sp = torch.sin(pitch / 2)
    cr = torch.cos(roll / 2)
    sr = torch.sin(roll / 2)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    return torch.stack([w, x, y, z], dim=-1)

def matrix_to_quaternion(rotation_matrix):
    """
    Convert a 3x3 rotation matrix to a unit quaternion.
    """
    if rotation_matrix.dim() == 2:
        rotation_matrix = rotation_matrix[None, ...]
    assert rotation_matrix.shape[1:] == (3, 3)

    # Compute traces manually instead of using vmap (for compatibility)
    traces = rotation_matrix[:, 0, 0] + rotation_matrix[:, 1, 1] + rotation_matrix[:, 2, 2]
    quaternion = torch.zeros(
        rotation_matrix.shape[0],
        4,
        dtype=rotation_matrix.dtype,
        device=rotation_matrix.device,
    )
    for i in range(rotation_matrix.shape[0]):
        matrix = rotation_matrix[i]
        trace = traces[i]
        if trace > 0:
            S = torch.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (matrix[2, 1] - matrix[1, 2]) / S
            y = (matrix[0, 2] - matrix[2, 0]) / S
            z = (matrix[1, 0] - matrix[0, 1]) / S
        elif (matrix[0, 0] > matrix[1, 1]) and (matrix[0, 0] > matrix[2, 2]):
            S = torch.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            w = (matrix[2, 1] - matrix[1, 2]) / S
            x = 0.25 * S
            y = (matrix[0, 1] + matrix[1, 0]) / S
            z = (matrix[0, 2] + matrix[2, 0]) / S
        elif matrix[1, 1] > matrix[2, 2]:
            S = torch.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            w = (matrix[0, 2] - matrix[2, 0]) / S
            x = (matrix[0, 1] + matrix[1, 0]) / S
            y = 0.25 * S
            z = (matrix[1, 2] + matrix[2, 1]) / S
        else:
            S = torch.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            w = (matrix[1, 0] - matrix[0, 1]) / S
            x = (matrix[0, 2] + matrix[2, 0]) / S
            y = (matrix[1, 2] + matrix[2, 1]) / S
            z = 0.25 * S

        quaternion[i] = torch.tensor(
            [w, x, y, z], dtype=matrix.dtype, device=matrix.device
        )
    return quaternion.squeeze()
