import torch
try:
    from gsplat.cuda._wrapper import spherical_harmonics
except ImportError:
    print("Please install gsplat>=1.0.0")

from ..utils.gaussian_utils import quat_mult, quat_to_rotmat
from .rigid_object import RigidPortableSubModel


def flip_spherical_harmonics(coeff):
    """
    Flip the spherical harmonics coefficients along the y-axis.

    Args:
        coeff (torch.Tensor): A tensor of shape [N, 16, 3], where N is the number of Gaussians,
                              16 is the number of spherical harmonics coefficients (up to degree l=3),
                              and 3 is the feature dimension.

    Returns:
        torch.Tensor: The flipped spherical harmonics coefficients.
    """
    # Indices corresponding to m < 0 for l up to 3
    indices_m_negative = [1, 4, 5, 9, 10, 11]

    # Create a flip factor tensor of ones and minus ones
    flip_factors = torch.ones(coeff.shape[1], device=coeff.device)
    flip_factors[indices_m_negative] = -1

    # Reshape flip_factors to [1, 16, 1] for broadcasting
    flip_factors = flip_factors.view(1, -1, 1)

    # Apply the flip factors to the coefficients
    flipped_coeff = coeff * flip_factors

    return flipped_coeff


class MirroredRigidPortableSubModel(RigidPortableSubModel):
    MODEL_TYPE = "mirrored"

    def get_means(self, global_quat, global_trans):
        local_means: torch.Tensor = self.gauss_params['means']
        local_means = local_means.unsqueeze(0).repeat(2, 1, 1)
        local_means[1, :, :] = local_means[1, :, :] * local_means.new_tensor([1, -1, 1]).view(1, 3)  # flip y
        local_means = local_means.view(-1, 3)

        rot_cur_frame = quat_to_rotmat(global_quat[None, ...])[0, ...]
        self.global_means = local_means @ rot_cur_frame.T + global_trans
        return self.global_means

    def get_quats(self, global_quat, global_trans=None):
        local_quats = self.quats / self.quats.norm(dim=-1, keepdim=True)
        local_quats = local_quats.unsqueeze(0).repeat(2, 1, 1)
        local_quats[1, :, :] = local_quats[1, :, :] * local_quats.new_tensor([1, -1, 1, -1]).view(1, 4)  # flip quats at y axis
        local_quats = local_quats.view(-1, 4)

        global_quats = quat_mult(global_quat[None, ...], local_quats)
        return global_quats

    def get_scales(self):
        return torch.exp(self.scales).repeat(2, 1)

    def get_gaussian_rgbs(self, camera_to_worlds, timestamp, device=None):
        device = device if device is not None else self.device
        assert device != torch.device("cpu"), "`sphereical_harmonics` in `gsplat` only supports CUDA"
        true_features_dc = self.get_true_features_dc(timestamp, None)
        colors = torch.cat((true_features_dc[:, None, :], self.features_rest), dim=1).to(device)
        colors = colors.unsqueeze(0).repeat(2, 1, 1, 1)
        colors[1, ...] = flip_spherical_harmonics(colors[1, ...])
        colors = colors.view(-1, 16, 3)
        if self.sh_degree > 0:
            viewdirs = self.global_means.detach().to(device) - camera_to_worlds[..., :3, 3].to(device)  # (C, N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            rgbs = spherical_harmonics(self.sh_degree, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])

        return rgbs

    def get_opacity(self):
        return torch.sigmoid(self.gauss_params['opacities']).squeeze(-1).repeat(2)
