from __future__ import annotations
import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Optional, Tuple, Type, Union, Any
from typing_extensions import Literal
from pathlib import Path

import numpy as np
import cv2
from pyquaternion import Quaternion
import torch

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

from src.utils.alpasim_utils.geometry_utils import Sim2
from src.render.base_renderer import BaseRenderer, RenderState
from src.render.src.utils.gaussian_utils import matrix_to_quaternion, quat_to_rotmat, quat_to_angle
from src.render.src.utils.portable_utils import convert_to_attribute_dict, AttrDict
from src.render.src.gaussian_model.vanilla_gaussian_splatting import VanillaPortableGaussianModel as VanillaModel
from src.render.src.gaussian_model.rigid_object import RigidPortableSubModel as RigidModel
from src.render.src.gaussian_model.rigid_object_mirrored import MirroredRigidPortableSubModel as MirroredModel

logger = logging.getLogger(__name__)

MODEL_MAPPING = {
    "VanillaGaussianSplattingModel": VanillaModel,
    "SkyboxGaussianSplattingModel": VanillaModel,
    "RigidSubModel": RigidModel,
    "MirroredRigidSubModel": MirroredModel,
    "DeformableSubModel": None,
}

def auto_submodel(atom_asset):
    if not isinstance(atom_asset, AttrDict):
        atom_asset = convert_to_attribute_dict(atom_asset)
    original_model_type = atom_asset.config.type
    return MODEL_MAPPING[original_model_type]


class MTGS(BaseRenderer):
    def __init__(
        self,
        *args,
        asset_folder_path: Union[str, Path],
        enable_collider: bool = False,
        render_depth: bool = False,
        rasterize_mode: Literal["classic", "antialiased"] = "classic",
        radius_clip: float = 0.,
        background_color: Union[Literal["random", "black", "white"], Tuple] = "black",
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.enable_collider = enable_collider
        self.render_depth = render_depth
        self.rasterize_mode = rasterize_mode
        self.radius_clip = radius_clip
        self.bg_color = self._init_background_color(background_color)
        self.asset_manager = MTGSAssetManager(Path(asset_folder_path), self.device)
        self.sensor_caches = None
        self.world_to_nre = None  

    def _init_background_color(self, background_color):
        if isinstance(background_color, str):
            if background_color == "random":
                background = torch.rand(3, device=self.device)
            elif background_color == "white":
                background = torch.ones(3, device=self.device)
            elif background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                raise ValueError(f"Unknown background color {background_color}")
            return background
        elif isinstance(background_color, tuple):
            assert len(background_color) == 3
            for color in background_color:
                assert 0 <= color <= 255
            return torch.tensor(background_color, device=self.device).float() / 255.

    @property
    def background_color(self) -> torch.Tensor:
        return self.bg_color.to(self.device)

    def _prepare_metas(self):
        self.model_types = {}
        self.node_types = {}
        self.submodel_names = {}
        self.timestamp = None
        self.sensor_mapping = {}
    
    def _init_gaussian_models(self):
        if hasattr(self, "gaussian_models"):
            del self.gaussian_models
            torch.cuda.empty_cache()
        self.gaussian_models = torch.nn.ParameterDict()

    @cached_property
    def background_asset(self):
        return self.gaussian_models["background"]

    @property
    def num_submodels(self):
        return len(self.node_types.keys())

    @property
    def num_cameras(self):
        return len(self.sensor_mapping.keys())

    def reset(self, current_scene_id, asset_id: Optional[str] = None, **kwargs):
        """
        Reset the renderer for a new scene.
        
        Args:
            current_scene_id: Scene ID
            asset_id: Optional asset ID. If None, uses current_scene_id
        """
        # Use provided asset_id or fall back to scene_id
        if asset_id is None:
            asset_id = current_scene_id

        reset_asset = self.asset_manager.reset(asset_id)
        if reset_asset:
            self._prepare_metas()
            self._init_gaussian_models()
            self.set_asset(self.asset_manager.background_asset_dict)
            # Note: calibrate_agent_state now requires ego2globals to be passed explicitly
            # This will be called separately with the required data
            self.sensor_caches = None
        
        # Reset world_to_nre when changing scenes
        self.world_to_nre = None
        
        # Mark as initialized for this scene
        self._scene_id = current_scene_id
        self._initialized = True
        logger.info(f"Renderer reset completed for scene {current_scene_id} (asset: {asset_id})")

    def set_world_to_nre(self, world_to_nre: np.ndarray):
        self.world_to_nre = world_to_nre
        logger.info(f"Set world_to_nre transformation: translation={world_to_nre[:3, 3]}")
    
    def calibrate_agent_state(self, ego2globals: Optional[torch.Tensor] = None):
        """
        Get the agent states in the reconstruction.
        
        Args:
            ego2globals: Optional array/tensor of shape [N, 4, 4] containing ego-to-global transformation matrices.
                        Can be numpy array or torch.Tensor. If None, will try to extract from loaded gaussian models.
        
        Returns:
            Dictionary mapping agent IDs to their states (translation and rotation)
        """
        mtgs_agent2states = {}
        
        # Handle ego state from ego2globals if provided
        if ego2globals is not None:
            # Convert to torch tensor if it's a numpy array
            if isinstance(ego2globals, np.ndarray):
                ego2globals_tensor = torch.tensor(ego2globals, dtype=torch.float64, device=self.device)
            else:
                # Already a torch tensor, just ensure correct dtype and device
                ego2globals_tensor = ego2globals.to(dtype=torch.float64, device=self.device)
            
            if ego2globals_tensor.ndim == 2:
                ego2globals_tensor = ego2globals_tensor.unsqueeze(0)  # Add batch dimension if needed
            
            # Extract translation and rotation from ego2globals
            # Directly use them without any transformation (matching WorldEngine's behavior)
            mtgs_ego2globals_trans = ego2globals_tensor[:, :3, 3]
            mtgs_ego2globals_quat = matrix_to_quaternion(ego2globals_tensor[:, :3, :3])
            
            mtgs_agent2states['ego'] = {
                'translation': mtgs_ego2globals_trans,
                'rotation': mtgs_ego2globals_quat,
            }

        # Extract agent states from loaded gaussian models
        for asset_token, model_name in self.submodel_names.items():
            if self.node_types[asset_token] != 'na':
                continue

            if self.gaussian_models[model_name].static_in_log:
                continue

            in_frame_mask = self.gaussian_models[model_name].log_trans[:, 2] < 1000

            mtgs_agent2states[asset_token] = {
                'translation': self.gaussian_models[model_name].log_trans[in_frame_mask].double(),
                'rotation': self.gaussian_models[model_name].log_quats[in_frame_mask].double(),
            }
        return mtgs_agent2states

    def get_submodel_name(self, token):
        node_type = self.node_types[token]
        if len(node_type) > 3:
            return token
        model_type = self.model_types[token]
        return f"{node_type}_{model_type}_{token}"
    
    def set_asset(self, asset):
        is_background = ("background" in asset.keys())
        if is_background:
            self.recon2global_translation = asset['background']['config']['recon2world_translation']

        asset_type = "background" if is_background else "foreground"
        for asset_name in asset.keys():
            logger.debug(f"Loading {asset_type} asset <{asset_name}>...")
            sub_asset = convert_to_attribute_dict(asset[asset_name])
            asset_class = auto_submodel(sub_asset)
            asset_token = asset_name.split("_")[-1]
            if asset_token in ["background", "skybox"]:
                self.node_types[asset_token] = asset_token
            else:
                self.node_types[asset_token] = "na" if is_background else "ia"
                # short for `native agent` and `inserted agent` respectively

            self.model_types[asset_token] = asset_class.MODEL_TYPE
            model_name = self.get_submodel_name(asset_token)
            self.submodel_names[asset_token] = model_name
            self.gaussian_models[model_name] = asset_class(
                asset=sub_asset,
                model_name=model_name
            )

    @staticmethod
    def get_transform_matrix(rotation, translation):
        if not isinstance(rotation, np.ndarray):
            rotation = rotation.cpu().numpy()
        if not isinstance(translation, np.ndarray):
            translation = translation.cpu().numpy()
        if rotation.ndim == 1:
            # is quaternion
            rotation = Quaternion(rotation).rotation_matrix
        if translation.ndim == 1:
            # change to column vector
            translation = np.expand_dims(translation, axis=-1)
        if rotation.ndim == 2:
            rotation = np.expand_dims(rotation, axis=0)
            translation = np.expand_dims(translation, axis=0)

        transform = np.tile(np.eye(4), (rotation.shape[0], 1, 1))
        transform[:, :3, :3] = rotation
        transform[:, :3, 3:4] = translation
        return torch.from_numpy(transform).float()

    def set_sensors(self, sensors, ego2global):
        if self.sensor_caches is not None:
            camera_to_egos, intrinsics, map_inverse_distorts, hw_dict = self.sensor_caches
            camera_to_worlds = torch.einsum("ij,bjk->bik", ego2global, camera_to_egos)
            return camera_to_worlds, intrinsics, map_inverse_distorts, hw_dict

        self.sensor_mapping = {}
        sensor2egos = []
        intrinsics = []
        map_inverse_distorts = []
        height, width = None, None
        for idx, (cam_name, cam_info) in enumerate(sensors.items()):
            self.sensor_mapping[cam_name] = idx
            sensor2ego = self.get_transform_matrix(
                rotation=cam_info["sensor2ego_rotation"], 
                translation=cam_info["sensor2ego_translation"]
            ).squeeze().to(self.device)             # 4 x 4

            intrinsic = np.array(cam_info["intrinsic"])
            distortion = np.array(cam_info["distortion"])
            new_intrinsic, roi = cv2.getOptimalNewCameraMatrix(
                intrinsic, distortion, (cam_info["width"], cam_info["height"]), 1
            )
            map_inverse_distort = cv2.initInverseRectificationMap(
                intrinsic, distortion, None, new_intrinsic, (cam_info["width"], cam_info["height"]), cv2.CV_32FC1
            )

            intrinsic_torch = torch.from_numpy(new_intrinsic).float().to(self.device).squeeze()
            sensor2egos.append(sensor2ego)
            intrinsics.append(intrinsic_torch)
            map_inverse_distorts.append(map_inverse_distort)

            assert (height is None) or (height == cam_info["height"]), "inconsistent height among RGB cameras"
            assert (width is None) or (width == cam_info["width"]), "inconsistent width among RGB cameras"
            height, width = cam_info["height"], cam_info["width"]

        sensor2egos = torch.stack(sensor2egos, dim=0)      # C x 4 x 4
        intrinsics = torch.stack(intrinsics, dim=0)     # C x 3 x 3
        height = height if isinstance(height, int) else height.item()
        width = width if isinstance(width, int) else width.item()
        self.sensor_caches = (sensor2egos, intrinsics, map_inverse_distorts, {"height": height, "width": width})

        camera_to_worlds = torch.einsum("ij,bjk->bik", ego2global, sensor2egos)
        return camera_to_worlds, intrinsics, map_inverse_distorts, {"height": height, "width": width}

    def update_world(self, timestamp, agent_states):
        
        if (self.timestamp == timestamp and 
            hasattr(self, "collected_gaussians") and 
            self.collected_gaussians and 
            'means' in self.collected_gaussians):
            return
        gs_dict = {
            "means": [],
            "scales": [],
            "quats": [],
            "opacities": [],
        }
        for asset_token in self.node_types.keys():
            gaussian_model = self.gaussian_models[self.submodel_names[asset_token]]
            if asset_token in agent_states.keys():
                quat, trans = self.get_agent_pose(asset_token, agent_states[asset_token])
            else:
                quat, trans = None, None            # use original log (for background, skybox, etc.)

            try:
                gs = gaussian_model.get_global_gaussians(
                    quat=quat,
                    trans=trans,
                    timestamp=timestamp,
                )
                if gs is None:
                    logger.warning(f"get_global_gaussians returned None for {asset_token}")
                    continue
                for k in gs_dict.keys():
                    gs_dict[k].append(gs[k].to(self.device))
            except Exception as e:
                logger.error(f"Error collecting gaussians for {asset_token}: {e}", exc_info=True)
                raise
        
        # Check if any gaussians were collected
        if not gs_dict["means"]:
            logger.error(f"No gaussians collected in update_world! node_types: {list(self.node_types.keys())}, agent_states: {list(agent_states.keys())}, timestamp={timestamp}")
            raise RuntimeError(f"No gaussians collected - no models available for timestamp {timestamp}")
        
        try:
            new_collected_gaussians = {}
            for key, value in gs_dict.items():
                new_collected_gaussians[key] = torch.cat(value, dim=0)
            self.collected_gaussians = new_collected_gaussians
            self.timestamp = timestamp
        except Exception as e:
            logger.error(f"Failed to merge gaussians: {e}", exc_info=True)
            raise

    def update_gaussian_rgbs(self, camera_to_worlds):
        rgb_list = []
        for asset_token in self.node_types.keys():
            gaussian_model = self.gaussian_models[self.submodel_names[asset_token]]
            rgbs = []
            for i in range(camera_to_worlds.shape[0]):
                rgbs.append(gaussian_model.get_gaussian_rgbs(
                        camera_to_worlds=camera_to_worlds[i:i+1],
                        timestamp=self.timestamp,
                        device=self.device
                    ).unsqueeze(0)
                )
            rgbs = torch.cat(rgbs, dim=0)       # C x ni x 3
            rgb_list.append(rgbs)
            if torch.isnan(rgbs).any() or torch.isinf(rgbs).any():
                print(f"NaN or inf in rgbs for model {self.submodel_names[asset_token]}")

        self.collected_gaussians['rgbs'] = torch.cat(rgb_list, dim=1)   # C x N x 3

    @torch.no_grad()
    def render(self, render_state: RenderState):
        timestamp = render_state[RenderState.TIMESTAMP]
        agent_states = render_state[RenderState.AGENT_STATE] # Simulator Local (Ego rear-axis)
        
        converted_agent_states = {}
        for agent_id, agent_state in agent_states.items():
            agent_state_np = np.array(agent_state, dtype=np.float64)
            
            # Step 1: Simulator Local → Nuplan Global
            if self.local2global_translation_xy is not None:
                agent_state_np[:2] += self.local2global_translation_xy
            
            # Step 2: Nuplan Global → Reconstruction Local 
            if isinstance(self.recon2global_translation, torch.Tensor):
                recon2global_xy = self.recon2global_translation[:2].cpu().numpy()
            else:
                recon2global_xy = np.array(self.recon2global_translation[:2])
            agent_state_np[:2] -= recon2global_xy
            
            converted_agent_states[agent_id] = agent_state_np

        self.update_world(
            timestamp=timestamp,
            agent_states=converted_agent_states,
        )
        ego2global_raw = self.get_agent_pose('ego', converted_agent_states['ego'], return_matrix=True)
        ego2global = ego2global_raw.to(device=self.device, dtype=torch.float32)

        cameras = render_state[RenderState.CAMERAS]
        camera_to_worlds, INTRINSICS, map_inverse_distorts, SHAPE = self.set_sensors(
            sensors=cameras,
            ego2global=ego2global,
        )
        self.update_gaussian_rgbs(camera_to_worlds)
        
        # shift the camera to center of scene looking at center
        R = camera_to_worlds[:, :3, :3]     # C x 3 x 3
        T = camera_to_worlds[:, :3, 3:4]    # C x 3 x 1
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.transpose(1,2)            # C x 3 x 3
        T_inv = -R_inv @ T
        viewmat = self.get_transform_matrix(
                rotation=R_inv, 
                translation=T_inv
            ).to(self.device)               # C x 4 x 4

        if self.render_depth:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        render, alpha, _ = rasterization(
            means=self.collected_gaussians['means'],
            quats=self.collected_gaussians['quats'],
            scales=self.collected_gaussians['scales'],
            opacities=self.collected_gaussians['opacities'],
            colors=self.collected_gaussians['rgbs'],
            viewmats=viewmat,       # [C, 4, 4]
            Ks=INTRINSICS.cuda(),   # [C, 3, 3]
            width=SHAPE['width'],
            height=SHAPE['height'],
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            radius_clip=self.radius_clip,
        )
        alpha = alpha[:, ...]
        rgb = render[:, ..., :3] + (1 - alpha) * self.background_color
        rgb = torch.clamp(rgb, 0.0, 1.0)
        rgb = rgb.cpu().numpy() * 255
        rgb = rgb.astype(np.uint8)
        rgb = [cv2.cvtColor(rgb[i], cv2.COLOR_RGB2BGR) for i in range(rgb.shape[0])]

        outputs = {"image": rgb}
        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max())
            outputs["depth"] = depth_im.cpu().numpy()

        sensor_dict = {}
        for cam_name, cam_idx in self.sensor_mapping.items():
            sensor_dict[cam_name] = {}
            for image_key in outputs.keys():
                sensor_dict[cam_name][image_key] = cv2.remap(
                    outputs[image_key][cam_idx],
                    map_inverse_distorts[cam_idx][0], 
                    map_inverse_distorts[cam_idx][1], cv2.INTER_LINEAR)
        return_dict = {
            'cameras': sensor_dict, 
            'lidars': {}    # LiDAR rendering is not supported
        }

        ego2global_nuplan = ego2global_raw.cpu().numpy()
        ego2global_nuplan[:3, 3] = ego2global_nuplan[:3, 3] + self.recon2global_translation[:3]

        meta_data_dict = {
            'ego2global': ego2global_nuplan,
            'render_state': render_state
        }
        return_dict.update(meta_data_dict)
        return return_dict

    def get_agent_pose(self, name, agent_state, return_matrix=False):
        if name not in self.mtgs_agent2states.keys():
            return None if return_matrix else (None, None)

        mtgs_agent_state = self.mtgs_agent2states[name]

        agent_state_xy = torch.tensor(agent_state[:2], device=self.device, dtype=torch.float64)
        agent_state_heading = torch.tensor(agent_state[-1], device=self.device, dtype=torch.float64)

        nearsest_idx = torch.argmin(torch.norm(mtgs_agent_state['translation'][:, :2] - agent_state_xy, dim=-1))
        log_trans = mtgs_agent_state['translation'][nearsest_idx]
        log_quat = mtgs_agent_state['rotation'][nearsest_idx]

        log_rot_matrix = quat_to_rotmat(log_quat)
        log_rot_yaw = quat_to_angle(log_quat, focus="yaw")["yaw"]
        heading_diff = agent_state_heading - log_rot_yaw
        cos_vals = torch.cos(heading_diff)
        sin_vals = torch.sin(heading_diff)
        rot_matrix_diff = torch.zeros(3, 3, device=self.device, dtype=torch.float64)
        rot_matrix_diff[0, 0] = cos_vals
        rot_matrix_diff[0, 1] = -sin_vals
        rot_matrix_diff[1, 0] = sin_vals
        rot_matrix_diff[1, 1] = cos_vals
        rot_matrix_diff[2, 2] = 1.0

        new_trans = log_trans.clone()
        new_trans[:2] = agent_state_xy
        new_rot_matrix = rot_matrix_diff @ log_rot_matrix

        if return_matrix:
            agent2global = torch.zeros(4, 4, device=self.device, dtype=torch.float64)
            agent2global[:3, :3] = new_rot_matrix
            agent2global[:3, 3] = new_trans
            return agent2global
        else:
            quat = matrix_to_quaternion(new_rot_matrix)
            trans = new_trans
            return quat, trans
    
    @property
    def local2global_translation_xy(self) -> Optional[np.ndarray]:
        if self.world_to_nre is None:
            return None
        return -self.world_to_nre[:2, 3]


class MTGSAssetManager:

    def __init__(self, asset_folder_path: Path, device: torch.device):
        self.asset_folder_path = asset_folder_path
        self.current_asset_id = None
        self.device = device

    def reset(self, asset_id: str):
        if asset_id[-4:].startswith('-'):  # Check if ends with pattern like "-001"
            asset_id = asset_id[:-4]  # Remove the suffix

        if self.current_asset_id == asset_id:
            return False

        # Clean up old assets
        if getattr(self, "background_asset_dict", None) is not None:
            del self.background_asset_dict
            self.background_asset_dict = None
            torch.cuda.empty_cache()

        self.current_asset_id = asset_id
        self.load_asset()
        return True

    def load_asset(self):
        """
        Folder structure:
            {asset_dir}
            ├── background
            ├── road_height_map
            └── video_scene_dict.pkl (contains ego2global, etc.)
        """
        self.asset_dir = Path(self.asset_folder_path) / self.current_asset_id
        logger.info(f"Loading assets from asset {self.current_asset_id}...")
        # load background asset (includes background, skybox, and rigid objects)
        background_asset_path = self.asset_dir / 'background' / f'{self.current_asset_id}.ckpt'
        self.background_asset_dict = torch.load(background_asset_path, map_location=self.device)
        logger.info(f"Loaded background asset dict with {len(self.background_asset_dict)} models")

        road_height_map_path = self.asset_dir / 'road_height_map'
        self.road_height_map = dict(
            map = np.load(road_height_map_path / 'road_height_map.npy'),
            sim2 = Sim2.from_json(road_height_map_path / 'sim2.json')
        )

        # Load video_scene_dict_final.pkl if available (contains ego2global, etc.)
        video_scene_dict_path = self.asset_dir / 'video_scene_dict.pkl'
        self.video_scene_dict = None
        if video_scene_dict_path.exists():
            try:
                import pickle
                with open(video_scene_dict_path, 'rb') as f:
                    self.video_scene_dict = pickle.load(f)
                logger.info(f"Loaded video_scene_dict from {video_scene_dict_path}")
            except Exception as e:
                logger.warning(f"Failed to load video_scene_dict.pkl: {e}")

        # TODO: load foreground assets.
        self.foreground_asset_dir = self.asset_dir / 'foreground'
        self.foreground_assets = list(self.foreground_asset_dir.glob('*.ckpt'))


# Backward compatibility aliases
DigitalTwin = MTGS
DigitalTwinAssetManager = MTGSAssetManager
