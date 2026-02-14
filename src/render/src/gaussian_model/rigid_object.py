import logging

import torch
from torch.nn import Parameter, Module
try:
    from gsplat.cuda._wrapper import spherical_harmonics
except ImportError:
    print("Please install gsplat>=1.0.0")

from ..utils.gaussian_utils import quat_mult, quat_to_rotmat, interpolate_quats, IDFT
from .vanilla_gaussian_splatting import VanillaPortableGaussianModel
logger = logging.getLogger(__name__)

class RigidPortableSubModel(VanillaPortableGaussianModel):
    """Portable Gaussian Splatting model

    Args:
        asset: portable Gaussian Model with config include
               - type
               - sh_degree
               - scale_dim
               - fourier_features_dim
               - fourier_features_scale
               - fourier_in_space
               - log_timestamps
    """
    MODEL_TYPE = "rigid"

    def __init__(self, **kwargs):
        self.log_replay = kwargs.get("log_replay", False)
        super().__init__(**kwargs)
        # TODO
        # self.visible_range = kwargs.get("visible_range", None)
    
    def _update_default_config(self, config):
        config = super()._update_default_config(config)
        config["fourier_features_dim"] = config.get("fourier_features_dim", None)
        config["fourier_features_scale"] = config.get("fourier_features_scale", 1.0)
        config["fourier_in_space"] = config.get("fourier_in_space", 'temporal')
        config["log_timestamps"] = config.get("log_timestamps", None)
        return config

    def load_state_dict(self, dict: dict):
        super().load_state_dict(dict)
        self.exist_log = ('instance_trans' in dict.keys())
        if self.exist_log:
            self.log_trans = Parameter(dict['instance_trans'].squeeze())
            self.log_quats = Parameter(dict['instance_quats'].squeeze())
            self.static_in_log = (self.log_trans.dim() == 1)
            if not self.static_in_log:
                assert getattr(self.config, "log_timestamps", None) is not None
                self.log_start_time = self.config.log_timestamps.min().item()
                self.log_timestamps = self.config.log_timestamps.squeeze() - self.log_start_time

                # TODO:
                # maybe make some transition trajectories at the beginning and the ending to avoid sudden appearance or disappearance
            
            log_type = "static" if self.static_in_log else "dynamic"
            logger.debug(f"Log pose for `{self.model_name_abbr}` loaded. LOG TYPE: `{log_type}`")
            if self.log_replay:
                logger.debug(f"Log replay for `{self.model_name_abbr}` enabled.")

    def set_static_force(self):
        self.static_in_log = True
        in_frame_mask = self.log_trans[:, 2] < 1000
        self.log_trans = Parameter(self.log_trans[in_frame_mask][0])
        self.log_quats = Parameter(self.log_quats[in_frame_mask][0])

    def get_means(self, global_quat, global_trans):
        local_means = self.gauss_params['means']
        rot_cur_frame = quat_to_rotmat(global_quat[None, ...])[0, ...]
        self.global_means = local_means @ rot_cur_frame.T + global_trans
        return self.global_means

    def get_quats(self, global_quat, global_trans=None):
        local_quats = self.quats / self.quats.norm(dim=-1, keepdim=True)
        global_quats = quat_mult(global_quat[None, ...], local_quats)
        return global_quats

    def get_fourier_features(self, x):
        scaled_x = x * self.config.fourier_features_scale
        input_is_normalized = (self.config.fourier_in_space == 'temporal')
        idft_base = IDFT(scaled_x, self.config.fourier_features_dim, input_is_normalized).to(self.device)
        return torch.sum(self.features_dc * idft_base[..., None], dim=1, keepdim=False)
    
    def get_true_features_dc(self, timestamp=None, cam_obj_yaw=None):
        if self.config.fourier_features_dim is None:
            return self.features_dc
        normalized_x = timestamp if self.config.fourier_in_space == 'temporal' else cam_obj_yaw
        assert normalized_x is not None
        return self.get_fourier_features(normalized_x)

    def get_gaussian_rgbs(self, camera_to_worlds, timestamp, device=None):
        device = device if device is not None else self.device
        assert device != torch.device("cpu"), "`sphereical_harmonics` in `gsplat` only supports CUDA"
        true_features_dc = self.get_true_features_dc(timestamp, None)
        colors = torch.cat((true_features_dc[:, None, :], self.features_rest), dim=1).to(device)
        if self.sh_degree > 0:
            viewdirs = self.global_means.detach().to(device) - camera_to_worlds[..., :3, 3].to(device)  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            rgbs = spherical_harmonics(self.sh_degree, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])

        return rgbs

    def get_opacity(self):
        return torch.sigmoid(self.gauss_params['opacities']).squeeze(-1)

    def _get_log_pose_from_timestamp(self, timestamp):
        self.log_timestamps = self.log_timestamps.to(self.device)
        relative_timestamp = timestamp - self.log_start_time

        diffs = relative_timestamp - self.log_timestamps
        prev_frame = torch.argmin(torch.where(diffs >= 0, diffs, float('inf')))
        next_frame = torch.argmin(torch.where(diffs <= 0, -diffs, float('inf')))

        if next_frame == prev_frame:
            # Timestamp exactly matches a frame, no interpolation needed
            return self.log_quats[next_frame], self.log_trans[next_frame], timestamp

        # Calculate interpolation factor
        t = (relative_timestamp - self.log_timestamps[prev_frame]) / (self.log_timestamps[next_frame] - self.log_timestamps[prev_frame])
        
        # Interpolate quaternions (using slerp) and translations
        quat_interp = interpolate_quats(self.log_quats[prev_frame], self.log_quats[next_frame], t).squeeze()
        trans_interp = torch.lerp(self.log_trans[prev_frame], self.log_trans[next_frame], t)

        return quat_interp, trans_interp, timestamp
    
    def _decide_global_pose(self, quat=None, trans=None, timestamp=None):
        if self.static_in_log:
            return self.log_quats, self.log_trans, timestamp        
        if quat is not None and trans is not None:
            assert quat.shape == (4,) and trans.shape == (3,)
            return quat.float(), trans.float(), timestamp
        return self._get_log_pose_from_timestamp(timestamp)

    def get_global_gaussians(self, quat=None, trans=None, timestamp=None, **kwargs):
        quat, trans, timestamp = self._decide_global_pose(
                                        quat=quat,
                                        trans=trans,
                                        timestamp=timestamp,
                                    )

        return {
            "means": self.get_means(global_trans=trans, global_quat=quat),
            "scales": self.get_scales(),
            "quats": self.get_quats(global_trans=trans, global_quat=quat),
            "opacities": self.get_opacity(),
        }
