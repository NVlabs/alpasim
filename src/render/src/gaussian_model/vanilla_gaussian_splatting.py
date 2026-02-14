from typing import Dict, List, Optional, Tuple, Type, Union, Any

import numpy as np
import torch
from torch.nn import Parameter, Module

try:
    from gsplat.cuda._wrapper import spherical_harmonics
except ImportError:
    print("Please install gsplat>=1.0.0")

class VanillaPortableGaussianModel(torch.nn.Module):
    """Portable Gaussian Splatting model

    Args:
        asset: portable Gaussian Model with config include
               - type
               - sh_degree
               - scale_dim
    """
    MODEL_TYPE = "vanilla"

    def __init__(
        self,
        asset: Any,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.eval()
        self.model_type = self.MODEL_TYPE
        self.model_name = model_name
        self.config = self._update_default_config(asset.config)
        self._init_gauss_params()
        self.load_state_dict(asset.state_dict)

    def _update_default_config(self, config):
        config["sh_degree"] = config.get("sh_degree", None)
        config["scale_dim"] = config.get("scale_dim", 3)
        return config

    def _init_gauss_params(self):
        self.gauss_params = torch.nn.ParameterDict({
            "means": None,
            "scales": None,
            "quats": None,
            "features_dc": None,
            "features_rest": None,
            "opacities": None,
        })

    def load_state_dict(self, dict: dict):
        for name in self.gauss_params.keys():
            if "gauss_params" in dict:
                self.gauss_params[name] = Parameter(dict["gauss_params"][name])
            else:
                self.gauss_params[name] = Parameter(dict[f"gauss_params.{name}"])

    def get_isotropic_quats(self, num_points):
        quats = torch.zeros(num_points, 4)
        quats[:, 0] = 1.0
        return quats.to(self.device)

    @property
    def device(self):
        return self.gauss_params["means"].device

    @property
    def sh_degree(self):
        return self.config.sh_degree
    
    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self) -> torch.nn.Parameter:
        return self.gauss_params["means"]

    @property
    def scales(self) -> torch.nn.Parameter:
        if self.config.scale_dim == 3:
            return self.gauss_params["scales"]
        elif self.config.scale_dim == 1:
            return self.gauss_params["scales"].repeat(1, 3)

    @property
    def quats(self) -> torch.nn.Parameter:
        if self.config.scale_dim == 3:
            return self.gauss_params["quats"]
        elif self.config.scale_dim == 1:
            return self.get_isotropic_quats(self.num_points)

    @property
    def features_dc(self) -> torch.nn.Parameter:
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self) -> torch.nn.Parameter:
        return self.gauss_params["features_rest"]

    @property
    def opacities(self) -> torch.nn.Parameter:
        return self.gauss_params["opacities"]

    @property
    def model_name_abbr(self):
        return self.model_name.split("_")[-1]

    def get_means(self):
        return self.gauss_params['means']

    def get_scales(self):
        return torch.exp(self.scales)

    def get_quats(self): 
        quats = self.quats
        return quats / quats.norm(dim=-1, keepdim=True)

    def get_opacity(self):
        return torch.sigmoid(self.gauss_params['opacities']).squeeze(-1)
    
    def get_gaussian_rgbs(self, camera_to_worlds, device=None, **kwargs):
        device = device if device is not None else self.device
        assert device != torch.device("cpu"), "`sphereical_harmonics` in `gsplat` only supports CUDA"
        assert self.features_dc.dim() == 2, \
            "if Fourier embedding is used in models derived from `VanillaPortableGaussianModel`, a new `get_gaussian_rgbs` function must be implemented"
        colors = torch.cat((self.features_dc[:, None, :], self.features_rest), dim=1).to(device)
        if self.sh_degree > 0:
            viewdirs = self.means.detach().to(device) - camera_to_worlds[..., :3, 3].to(device)  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            rgbs = spherical_harmonics(self.sh_degree, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])
        
        return rgbs

    def get_global_gaussians(self, **kwargs):

        return {
            "means": self.get_means(),
            "scales": self.get_scales(),
            "quats": self.get_quats(),
            "opacities": self.get_opacity(),
        }

    def get_gaussian_params(self, **kwargs):
        raise ValueError("Should not be called.")

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        raise ValueError("Should not be called.")

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        raise ValueError("Should not be called.")

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)