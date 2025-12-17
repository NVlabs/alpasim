"""Configuration schema for driver service supporting multiple model backends."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from omegaconf import MISSING


class ModelType(str, Enum):
    """Supported model types."""

    VAM = "vam"
    TRANSFUSER = "transfuser"
    ALPAMAYO_R1 = "alpamayo_r1"


@dataclass
class ModelConfig:
    """Unified model configuration for all model types.

    Note: dtype is not exposed in config - each model hardcodes its expected dtype.
    VAM uses float16, Transfuser uses its TrainingConfig's torch_float_type.
    """

    model_type: ModelType = MISSING  # Type of model to use (VAM or TRANSFUSER)
    checkpoint_path: str = MISSING  # Path to model checkpoint (.pt/.pth)
    device: str = MISSING  # Device to run inference on (cuda/cpu)
    tokenizer_path: Optional[str] = None  # Only required for VAM


@dataclass
class InferenceConfig:
    """Inference configuration."""

    use_cameras: list[str] = MISSING
    max_batch_size: int = MISSING  # Maximum batch size for inference
    subsample_factor: int = 1
    context_length: Optional[int] = None  # Override model's default context length


@dataclass
class RouteConfig:
    """Route and command configuration."""

    default_command: int = 2  # Default command: 0=right, 1=left, 2=straight
    use_waypoint_commands: bool = True  # Whether to interpret waypoints as commands
    command_distance_threshold: float = (
        2.0  # Lateral displacement threshold for command determination (meters)
    )
    min_lookahead_distance: float = (
        5.0  # Minimum distance to look ahead for waypoints (meters)
    )


@dataclass
class TrajectoryOptimizerConfig:
    """Trajectory optimization configuration. (VaVam-Eco)"""

    enabled: bool = False  # Whether to enable trajectory optimization

    # Optimization weights
    smoothness_weight: float = 1.0  # Weight for trajectory smoothness
    deviation_weight: float = 0.1  # Weight for deviation from original
    comfort_weight: float = 2.0  # Weight for comfort constraint penalty

    max_iterations: int = 100  # Maximum optimization iterations

    # Frenet retiming options
    retime_in_frenet: bool = True  # Whether to redistribute waypoints along path
    retime_alpha: float = 0.25  # Retiming strength [0,1]; higher = more front-loaded

    # Vehicle constraints
    max_deviation: float = 2.0  # Max deviation from original trajectory (meters)
    max_heading_change: float = 0.5236  # Max heading change (~30 degrees)
    max_speed: float = 15.0  # Maximum speed (m/s)
    max_accel: float = 5.0  # Maximum acceleration (m/s²)

    # Comfort limits (from PDMS spec)
    max_abs_yaw_rate: float = 0.95  # rad/s
    max_abs_yaw_acc: float = 1.93  # rad/s²
    max_lon_acc_pos: float = 4.89  # m/s²
    max_lon_acc_neg: float = -4.05  # m/s²
    max_abs_lon_jerk: float = 8.37  # m/s³


@dataclass
class RectificationTargetConfig:
    """Target pinhole parameters for rectifying a rendered camera."""

    focal_length: tuple[float, float]
    principal_point: tuple[float, float]
    resolution_hw: tuple[int, int]
    radial: tuple[float, ...] = ()
    tangential: tuple[float, ...] = ()
    thin_prism: tuple[float, ...] = ()

    # We rectify a larger canvas and only crop in the end to allow for
    # margin when applying the distortion of the pinhole camera.
    max_overscan_scale: float = 2.0
    safety_margin_px: int = 10


@dataclass
class DriverConfig:
    """Main driver configuration supporting multiple model backends."""

    # Logging level (DEBUG, INFO, WARNING, ERROR)
    log_level: str = "INFO"

    # Model configuration
    model: ModelConfig = MISSING

    # Server configuration
    host: str = MISSING
    port: int = MISSING

    # Inference configuration
    inference: InferenceConfig = MISSING

    route: RouteConfig = field(default_factory=RouteConfig)

    trajectory_optimizer: TrajectoryOptimizerConfig = field(
        default_factory=TrajectoryOptimizerConfig
    )

    # Output configuration
    output_dir: str = MISSING

    # If true, generates debug images in `output_dir`
    plot_debug_images: bool = False

    # Optional per-camera rectification definitions
    rectification: Optional[dict[str, RectificationTargetConfig]] = None
