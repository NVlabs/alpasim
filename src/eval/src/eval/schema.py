# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from omegaconf import MISSING


class EgoLoc(StrEnum):
    # Center of the ego vehicle in the BEV view.
    CENTER = "center"
    # Position of the ego vehicle at the bottom center of the BEV view.
    # If you choose this, make sure to set `rotate_map_to_ego` to `True`.
    BOTTOM_CENTER = "bottom_center"


class MapElements(StrEnum):
    ROAD_LANE_CENTER = "road_lane_center"
    ROAD_LANE_LEFT_EDGE = "road_lane_left_edge"
    ROAD_LANE_RIGHT_EDGE = "road_lane_right_edge"
    ROAD_EDGE = "road_edge"
    STOP_LINE = "stop_line"
    OTHER_LINE = "other_line"
    # The ground truth trajectory as line on the road.
    GT_LINESTRING = "gt_linestring"
    # A "ghost" car driving the ground truth trajectory.
    EGO_GT_GHOST_POLYGON = "ego_gt_ghost_polygon"
    # Selected and sampled trajectories.
    DRIVER_RESPONSES = "driver_responses"
    # The route passed to the EGO.
    ROUTE = "route"
    # All other agents in the scene.
    AGENTS = "agents"


class VideoLayout(StrEnum):
    # Default debug view: BEV map, camera, metrics.
    DEFAULT = "default"
    # Reasoning overlay view: first-person camera with reasoning text overlay
    #   + trajectory chart on right.
    REASONING_OVERLAY = "reasoning_overlay"


@dataclass
class MapVideoConfig:
    # Radius around ego.
    map_radius_m: float = MISSING
    # Where to place the ego in the map.
    ego_loc: EgoLoc = MISSING
    # Whether to rotate the map to the ego's yaw (forward = going up).
    rotate_map_to_ego: bool = MISSING
    # List of map elements to plot. If None, all elements will be plotted.
    map_elements_to_plot: list[MapElements] | None = None


@dataclass
class VideoRendererConfig:
    # Whether to render the video at all
    render_video: bool = MISSING
    # List of video layouts to render.
    video_layouts: list[VideoLayout] = MISSING
    # Which camera to render (logical id)
    camera_id_to_render: str = MISSING
    # Whether to overlay planner trajectories onto the camera view
    overlay_plans_on_camera: bool = True
    # Order and inclusion of metrics in the rendered table; None shows all metrics.
    metrics_table_entries: list[str] | None = None
    # Options for how to render the BEV map
    map_video: MapVideoConfig = MISSING
    # Only render every nth frame
    render_every_nth_frame: int = MISSING
    # Whether to generate a combined video from all the videos
    generate_combined_video: bool = MISSING
    # Speed factor for ffmpeg for the combined video (e.g. 0.5 = double speed)
    combined_video_speed_factor: float = MISSING
    # How often to refresh the reasoning text (in seconds). If None, refresh every frame.
    reasoning_text_refresh_interval_s: float = MISSING


class MinADEScorerTarget(StrEnum):
    # Compare to the trajectory taken during simulation. This makes sense if the
    # ego is not log following and can give an indication of how well the model
    # followed through on its plans.
    SELF = "self"
    # Compare to the ground truth trajectory. This only really makes sense if
    # the ego is log following.
    GT = "gt"


@dataclass
class MinADEScorerConfig:
    # Time deltas to compute the minADE for.
    time_deltas: list[float] = MISSING
    # Whether to include the z-coordinate in the computation.
    incl_z: bool = MISSING
    # Whether to compute the minADE compared to the ground truth or to the
    # trajectory taken during simulation.
    target: MinADEScorerTarget = MISSING


@dataclass
class PlanDeviationScorerConfig:
    incl_z: bool = MISSING
    # We use exponential decay to weight the contribution sooner timesteps more
    # than later timesteps when aggregating over one plan.
    avg_decay_rate: float = MISSING
    # Minimum number of timesteps to consider for plan consistency. If the
    # number of overlapping timesteps between prev. and current plan is less
    # than this, the plan consistency not be computed.
    min_timesteps: int = MISSING


@dataclass
class ImageScorerConfig:
    camera_logical_id: str = MISSING


@dataclass
class ScorersConfig:
    min_ade: MinADEScorerConfig = MISSING
    plan_deviation: PlanDeviationScorerConfig = MISSING
    image: ImageScorerConfig = MISSING


@dataclass
class MetricVehicleConfig:
    # Whether to round the shapely vehicle corners. Impacts video and metrics.
    # 0.0 is no rounding, 1.0 is half the width of the vehicle, i.e. the corner
    # radius is vehicle_corner_roundness * min_vehicle_side_length / 2.
    vehicle_corner_roundness: float = MISSING
    # 0.0 = no shrinkage, 1.0 = reduced to a point.
    vehicle_shrink_factor: float = MISSING


@dataclass
class MetricAggregationModifiersConfig:
    # Maximum distance to the ground truth trajectory to consider for metrics.
    max_dist_to_gt_trajectory: float = MISSING


@dataclass
class EvalConfig:
    # Configuration for scorers that have free parameters.
    scorers: ScorersConfig = MISSING
    aggregation_modifiers: MetricAggregationModifiersConfig = MISSING
    # Number of processes to use for parallel processing of ASL reading and
    # metric computation.
    num_processes: int = MISSING
    # Whether to render a video, what should be rendered and how.
    video: VideoRendererConfig = MISSING
    # Vector map params. Passed to trajdata.
    vec_map: dict[str, Any] = MISSING
    vehicle: MetricVehicleConfig = MISSING
