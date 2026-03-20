# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Defines the omegaconf .yaml configuration format for Alpasim runtime.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Type, TypeVar, cast

from alpasim_utils.scenario import VehicleConfig
from alpasim_utils.yaml_utils import load_yaml_dict
from omegaconf import MISSING, OmegaConf

C = TypeVar("C")


@dataclass
class USDZSourceConfig:
    """Configuration for USDZ data source.

    Attributes:
        enabled: Whether this data source is enabled
        data_dir: Path to directory containing USDZ files
        desired_dt: Desired time delta between trajectory frames in seconds
        incl_vector_map: Whether to load vector maps (roads, lanes, etc.)
        asset_base_path: Base path for rendering assets (e.g., MTGS assets)
    """

    enabled: bool = True
    data_dir: str = MISSING
    desired_dt: float = 0.1  # 10 Hz sampling
    incl_vector_map: bool = True
    asset_base_path: Optional[str] = None


@dataclass
class NuPlanSourceConfig:
    """Configuration for NuPlan data source.

    Attributes:
        enabled: Whether this data source is enabled
        data_dir: Path to NuPlan dataset directory
        config_dir: Directory containing YAML scene config files for batch mode
        num_timesteps_before: Number of timesteps before central token (batch mode)
        num_timesteps_after: Number of timesteps after central token (batch mode)
        desired_dt: Desired time delta between frames in seconds
        incl_vector_map: Whether to load vector maps
    """

    enabled: bool = False
    data_dir: Optional[str] = None
    config_dir: Optional[str] = None
    num_timesteps_before: int = 30
    num_timesteps_after: int = 80
    desired_dt: float = 0.1
    incl_vector_map: bool = True


@dataclass
class DataSourceConfig:
    """Configuration for unified data loading through trajdata.

    This provides a hierarchical structure where common configuration is separated
    from data source-specific settings, making it easier to understand which
    parameters affect which data sources.

    Attributes:
        cache_location: Path to shared trajdata cache directory
        rebuild_cache: Whether to force rebuild the cache for all sources
        rebuild_maps: Whether to force rebuild maps for all sources
        num_workers: Number of parallel workers for cache creation
        usdz: USDZ-specific configuration
        nuplan: NuPlan-specific configuration
    """

    # Common configuration (applies to all data sources)
    cache_location: str = MISSING
    rebuild_cache: bool = False
    rebuild_maps: bool = False
    num_workers: int = 4

    # Source-specific configurations
    usdz: Optional[USDZSourceConfig] = None
    nuplan: Optional[NuPlanSourceConfig] = None

    def to_trajdata_params(self) -> dict:
        """Convert hierarchical config to flat parameters for trajdata's UnifiedDataset.

        Returns:
            Dictionary with keys expected by UnifiedDataset constructor

        Raises:
            ValueError: If no data sources are enabled
        """
        desired_data = []
        data_dirs = {}

        # Collect enabled sources
        if self.usdz is not None and self.usdz.enabled:
            desired_data.append("usdz")
            data_dirs["usdz"] = self.usdz.data_dir

        if self.nuplan is not None and self.nuplan.enabled:
            desired_data.append("nuplan")
            data_dirs["nuplan"] = self.nuplan.data_dir

        if not desired_data:
            raise ValueError("No data sources enabled in configuration")

        # Use first enabled source for common parameters
        # (desired_dt, incl_vector_map are typically consistent across sources)
        primary_source = self.usdz if (self.usdz and self.usdz.enabled) else self.nuplan

        params = {
            "desired_data": desired_data,
            "data_dirs": data_dirs,
            "cache_location": self.cache_location,
            "rebuild_cache": self.rebuild_cache,
            "rebuild_maps": self.rebuild_maps,
            "num_workers": self.num_workers,
            "desired_dt": primary_source.desired_dt,
            "incl_vector_map": primary_source.incl_vector_map,
        }

        # Add source-specific parameters
        if self.nuplan and self.nuplan.enabled and self.nuplan.config_dir:
            params["config_dir"] = self.nuplan.config_dir
            params["num_timesteps_before"] = self.nuplan.num_timesteps_before
            params["num_timesteps_after"] = self.nuplan.num_timesteps_after

        return params


def typed_parse_config(path: str | Path, config_type: Type[C]) -> C:
    """Reads a yaml file at `path` and parses it into a provided type using omegaconf."""
    yaml_config = OmegaConf.create(load_yaml_dict(path))

    schema = OmegaConf.structured(config_type)
    config: C = cast(C, OmegaConf.merge(schema, yaml_config))
    return config


@dataclass
class SingleUserEndpointConfig:
    """
    Configuration parameters for an endpoint *set by the user* - in contrast to
    addresses which are usually autogenerated by wizard.
    """

    skip: bool = False
    n_concurrent_rollouts: int = MISSING


@dataclass
class EndpointAddresses:
    """A list of URLs for a single endpoint, usually autogenerated by wizard."""

    addresses: list[str] = MISSING


@dataclass
class NetworkSimulatorConfig:
    """
    Addresses of all endpoints in simulation - this section is usually autogenerated by
    wizard.
    """

    sensorsim: EndpointAddresses = MISSING
    driver: EndpointAddresses = MISSING
    physics: EndpointAddresses = MISSING
    trafficsim: EndpointAddresses = MISSING
    controller: EndpointAddresses = MISSING


@dataclass
class RuntimeCameraConfig:
    """Configuration for a camera in the runtime. See `RuntimeCamera` for more details."""

    logical_id: str = "camera_front_wide_120fov"
    height: int = 160
    width: int = 256
    frame_interval_us: int = 33_000  # about 30fps
    shutter_duration_us: int = 17_000
    first_frame_offset_us: int = 0


@dataclass
class PoseConfig:
    translation_m: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


@dataclass
class OpenCVPinholeConfig:
    """OpenCV pinhole parameters (matches sensorsim OpenCVPinholeCameraParam)."""

    focal_length: tuple[float, float]
    principal_point: tuple[float, float]
    radial: tuple[float, ...] = ()  # k1..k6
    tangential: tuple[float, ...] = ()  # p1, p2
    thin_prism: tuple[float, ...] = ()  # s1..s4


@dataclass
class OpenCVFisheyeConfig:
    focal_length: tuple[float, float]
    principal_point: tuple[float, float]
    radial: tuple[float, ...] = ()  # k1..k4
    max_angle: Optional[float] = None


@dataclass
class LinearCdeConfig:
    linear_c: float
    linear_d: float
    linear_e: float


@dataclass
class FthetaConfig:
    principal_point: tuple[float, float]
    reference_poly: str  # Literal["pixel_to_angle", "angle_to_pixel"]
    pixeldist_to_angle: list[float] = field(default_factory=list)
    angle_to_pixeldist: list[float] = field(default_factory=list)
    max_angle: Optional[float] = None
    linear_cde: Optional[LinearCdeConfig] = None


@dataclass
class CameraIntrinsicsConfig:
    model: str  # Literal["opencv_pinhole", "opencv_fisheye", "ftheta"]
    opencv_pinhole: Optional[OpenCVPinholeConfig] = None
    opencv_fisheye: Optional[OpenCVFisheyeConfig] = None
    ftheta: Optional[FthetaConfig] = None


@dataclass
class CameraDefinitionConfig:
    """Configuration for overriding camera definitions from sensorsim.

    Only `logical_id` is required - it identifies which camera to override.
    Other fields are optional; if not provided, the values from sensorsim
    are preserved.
    """

    logical_id: str
    # For NRE sensorsim, logical_id must match one of the existing cameras:
    #     - 'camera_cross_left_120fov'
    #     - 'camera_cross_right_120fov'
    #     - 'camera_front_tele_30fov'
    #     - 'camera_front_wide_120fov'
    rig_to_camera: Optional[PoseConfig] = None
    intrinsics: Optional[CameraIntrinsicsConfig] = None
    resolution_hw: Optional[tuple[int, int]] = None
    # ShutterType is one of: [
    #     "ROLLING_TOP_TO_BOTTOM",
    #     "ROLLING_LEFT_TO_RIGHT",
    #     "ROLLING_BOTTOM_TO_TOP",
    #     "ROLLING_RIGHT_TO_LEFT",
    #     "GLOBAL"
    # ]
    shutter_type: Optional[str] = None


class PhysicsUpdateMode(Enum):
    NONE = 0
    EGO_ONLY = 1
    ALL_ACTORS = 2


class RouteGeneratorType(Enum):
    MAP = 0
    RECORDED = 1


@dataclass
class SimulationConfig:
    """
    Shared simulation parameters — applies to all scenes.
    """

    n_sim_steps: int = MISSING
    n_rollouts: int = MISSING

    control_timestep_us: int = 100_000
    pose_reporting_interval_us: int = 0  # 0 = no intermediate reporting
    force_gt_duration_us: int = 500_000  # 0.5s
    time_start_offset_us: int = (
        250_000  # 0.25s: there are often weird artifacts at the very start of a scene
    )

    # if true, we assert that each call to policy happens immediately after a frame has been
    # provided for each camera and the latest egomotion update. This flag does not modify
    # any parameters, only verifies that the config satisfies this condition (no user misconfiguration).
    # for extra information see the README.
    assert_zero_decision_delay: bool = False

    cameras: list[RuntimeCameraConfig] = field(
        default_factory=lambda: [RuntimeCameraConfig()]
    )

    # if None, the data will be pulled from the .usdz file
    vehicle: Optional[VehicleConfig] = None

    physics_update_mode: PhysicsUpdateMode = PhysicsUpdateMode.NONE

    image_format: str = "jpeg"  # Literal["jpeg", "png"]
    # Rig/directory in ego-hoods to use for ego masking. If None, no ego masking will be done.
    ego_mask_rig_config_id: Optional[str] = None
    planner_delay_us: int = (
        0  # models time delays from image capture to planner output to controller
    )

    route_generator_type: RouteGeneratorType = RouteGeneratorType.MAP

    # Whether to send optional messages to the driver
    send_recording_ground_truth: bool = False

    # Actors that appear for less than this amount of time will be dropped
    # before the simulation starts. Set to 0 to disable filtering.
    min_traffic_duration_us: int = 3_000_000  # 3 s

    # Flag to enable grouping of render requests
    group_render_requests: bool = False


@dataclass
class SceneConfig:
    """Scene to simulate and number of rollouts to run for it."""

    scene_id: str = MISSING
    n_rollouts: Optional[int] = None  # None = use SimulationConfig.n_rollouts


@dataclass
class UserEndpointConfig:
    """User-provided configuration for each of the endpoints"""

    sensorsim: SingleUserEndpointConfig = MISSING
    driver: SingleUserEndpointConfig = MISSING
    physics: SingleUserEndpointConfig = MISSING
    trafficsim: SingleUserEndpointConfig = MISSING
    controller: SingleUserEndpointConfig = MISSING

    sensorsim_cache_size: Optional[int] = None  # Scene cache size for sensorsim
    startup_timeout_s: int = 2 * 60  # 2 minutes by default
    do_shutdown: bool = True


@dataclass
class UserSimulatorConfig:
    """The section of simulator config created manually by the user"""

    simulation_config: SimulationConfig = MISSING
    scenes: list[SceneConfig] = MISSING
    enable_autoresume: bool = False
    endpoints: UserEndpointConfig = MISSING

    smooth_trajectories: bool = True  # whether to smooth trajectories with cubic spline
    extra_cameras: list[CameraDefinitionConfig] = field(default_factory=list)

    # Number of worker processes for parallel rollout execution.
    # 1 = inline mode, all in one process, good for debugging
    # >1 = multi-worker mode with subprocess-based parallelism
    nr_workers: int = MISSING

    # Unified data source configuration (required)
    # Data loading goes through trajdata's UnifiedDataset
    data_source: DataSourceConfig = MISSING


@dataclass
class SimulatorConfig:
    """
    The entire simulator config consists of two parts, one provided by the user manually
    (`user`) and `network` which is likely autogenerated by the deployment system like wizard.
    """

    user: UserSimulatorConfig = MISSING
    network: NetworkSimulatorConfig = MISSING
