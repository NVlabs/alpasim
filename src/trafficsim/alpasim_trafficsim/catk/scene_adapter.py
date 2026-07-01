# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import copy
from pathlib import Path

import numpy as np
import torch
from alpasim_trafficsim.catk.map_adapter import (
    ENVDATA_MAP_KEY_TO_CATK_MAP_ELEMENT_NAME,
    MAP_ELEMENT_NAME2_TYPEID,
    build_env_map_from_vector_map,
)
from alpasim_trafficsim.catk.obstacle_classes import (
    OBSTACLE_CLASS_NAME_TO_ID,
    obstacle_class_metadata,
)

# Runtime map layers that are not currently provided by the VectorMap adapter
# are null-filled so downstream consumers can rely on stable map keys.
_MAP_NULL_KEYS = (
    "poles",
    "traffic_signs",
    "traffic_lights",
    "nds_polygons",
    "road_markings",
)


class CATKSceneAdapter:
    """Convert artifact-backed scenes into CATK runtime dict state.

    Runtime uses a plain dict state with the existing keys expected by simulation
    and CATK prediction write-back. Scene content comes from the shared
    ``Artifact``/``SceneDataSource`` abstraction instead of reading scene files
    directly in trafficsim.
    """

    def __init__(
        self,
        num_history_steps: int = 16,
        motion_stepsize: float = 0.1,
        map_distance_x: float = 0,
        map_distance_y: float = 0,
        map_polyline_length_k: int = 1,
        map_resample_interval_m: float | None = 1.0,
        cache_size: int | None = None,
    ):
        if map_resample_interval_m is not None and map_resample_interval_m <= 0:
            raise ValueError(
                "map_resample_interval_m must be positive when provided, "
                f"got {map_resample_interval_m}"
            )
        self.motion_stepsize = motion_stepsize
        self._num_history_steps = num_history_steps
        self._map_distance_x = map_distance_x
        self._map_distance_y = map_distance_y
        self._map_polyline_length_k = map_polyline_length_k
        self._map_resample_interval_m = map_resample_interval_m
        self._cache: dict[str, dict] = {}
        self._cache_size = cache_size

    def load(self, data_source) -> dict:
        """Load CATK env data from an artifact-like scene data source.

        The generic SceneDataSource protocol exposes rig, traffic object, map,
        and metadata properties. The CATK adapter converts those scene-level
        abstractions into the mutable tensor dict used by the service.
        """
        if not all(
            hasattr(data_source, attr)
            for attr in ("rig", "traffic_objects", "map", "metadata")
        ):
            raise TypeError(
                "CATKSceneAdapter requires a scene data source exposing rig, "
                "traffic_objects, map, and metadata."
            )
        source_path = getattr(data_source, "source", "")
        scene_id = str(getattr(data_source, "scene_id", Path(str(source_path)).stem))
        cache_key = f"{scene_id}:{source_path}"
        if cache_key in self._cache:
            return copy.deepcopy(self._cache[cache_key])

        env_data = self._build_env_data_from_artifact(data_source, scene_id=scene_id)
        self._remember(cache_key, env_data)
        return copy.deepcopy(env_data)

    def _remember(self, cache_key: str, env_data: dict) -> None:
        if self._cache_size == 0:
            return
        if self._cache_size is not None and len(self._cache) >= self._cache_size:
            oldest_key = next(iter(self._cache))
            self._cache.pop(oldest_key)
        self._cache[cache_key] = env_data

    def _build_env_data_from_artifact(
        self,
        artifact,
        scene_id: str | None = None,
    ) -> dict:
        curr_t = self._num_history_steps - 1
        dt_us = int(round(self.motion_stepsize * 1_000_000))
        timestamps_us = _regular_timestamps_us(artifact.rig.trajectory, dt_us=dt_us)
        curr_t = min(curr_t, len(timestamps_us) - 1)
        t0_us = int(timestamps_us[curr_t])

        ego = _build_ego_from_rig(artifact.rig, timestamps_us)
        map_data = _build_map(
            artifact.map,
            ego=ego,
            curr_t=curr_t,
            distance_x=self._map_distance_x,
            distance_y=self._map_distance_y,
            map_polyline_length_k=self._map_polyline_length_k,
            map_resample_interval_m=self._map_resample_interval_m,
        )

        final_scene_id = (
            scene_id if scene_id is not None else artifact.metadata.scene_id
        )

        metadata = {
            **obstacle_class_metadata(),
            "scene_id": final_scene_id,
            "frame_rate": int(1.0 / self.motion_stepsize),
            "t0_us": t0_us,
            "map_source": "trajdata_vector_map",
        }

        agents, agent_object_ids, agent_is_static = _build_agents_from_traffic_objects(
            artifact.traffic_objects,
            timestamps_us,
        )

        env_key = {
            "curr_t": curr_t,
            "frame_rate": int(1.0 / self.motion_stepsize),
            "agent_object_ids": agent_object_ids,
            "agent_is_static": agent_is_static,
        }

        return {
            "metadata": metadata,
            "map": map_data,
            "agents": agents,
            "ego": ego,
            "env": env_key,
        }


def _build_map(
    vector_map,
    *,
    ego: dict,
    curr_t: int,
    distance_x: float,
    distance_y: float,
    map_polyline_length_k: int,
    map_resample_interval_m: float | None,
) -> dict:
    """Build runtime map layers from a trajdata VectorMap."""
    if vector_map is None:
        env_map = {
            env_key: None for env_key in ENVDATA_MAP_KEY_TO_CATK_MAP_ELEMENT_NAME
        }
    else:
        env_map = build_env_map_from_vector_map(
            vector_map,
            ego_xyz=ego["xyz"][curr_t],
            ego_heading=ego["heading"][curr_t],
            distance_x=distance_x,
            distance_y=distance_y,
            map_polyline_length_k=map_polyline_length_k,
            map_resample_interval_m=map_resample_interval_m,
        )

    for key in _MAP_NULL_KEYS:
        env_map[key] = None

    return env_map


def _regular_timestamps_us(trajectory, *, dt_us: int) -> np.ndarray:
    timestamps = np.asarray(trajectory.timestamps_us, dtype=np.int64)
    if timestamps.size == 0:
        raise ValueError("Artifact rig trajectory has no timestamps")
    start_us = int(timestamps[0])
    end_us = int(timestamps[-1])
    if end_us <= start_us:
        end_us = int(timestamps.max())
    if end_us <= start_us:
        return np.asarray([start_us], dtype=np.int64)
    steps = int(np.floor((end_us - start_us) / dt_us)) + 1
    return start_us + (np.arange(steps, dtype=np.int64) * dt_us)


def _trajectory_pose_tensors(
    trajectory, timestamps_us: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    xyz = torch.zeros((len(timestamps_us), 3), dtype=torch.float32)
    heading = torch.zeros((len(timestamps_us),), dtype=torch.float32)
    for idx, timestamp_us in enumerate(timestamps_us):
        pose = trajectory.interpolate_pose(int(timestamp_us))
        xyz[idx] = torch.as_tensor(pose.vec3, dtype=torch.float32)
        heading[idx] = float(pose.yaw())
    return xyz, heading


def _build_ego_from_rig(rig, timestamps_us: np.ndarray) -> dict:
    xyz, heading = _trajectory_pose_tensors(rig.trajectory, timestamps_us)
    vehicle_config = rig.vehicle_config
    if vehicle_config is None:
        lwh = torch.tensor([5.393, 2.109, 1.503], dtype=torch.float32)
    else:
        lwh = torch.tensor(
            [
                float(vehicle_config.aabb_x_m),
                float(vehicle_config.aabb_y_m),
                float(vehicle_config.aabb_z_m),
            ],
            dtype=torch.float32,
        )
    return {
        "xyz": xyz,
        "heading": heading,
        "lwh": lwh,
    }


def _build_agents_from_traffic_objects(
    traffic_objects,
    timestamps_us: np.ndarray,
) -> tuple[dict, list[str], list[bool]]:
    if not traffic_objects:
        return _build_empty_agents_for_steps(len(timestamps_us)), [], []

    objects = list(traffic_objects.values())
    num_agents = len(objects)
    num_steps = len(timestamps_us)
    xyz = torch.zeros((num_agents, num_steps, 3), dtype=torch.float32)
    heading = torch.zeros((num_agents, num_steps), dtype=torch.float32)
    valid_mask = torch.zeros((num_agents, num_steps), dtype=torch.bool)
    lwh = torch.zeros((num_agents, 3), dtype=torch.float32)
    track_ids = torch.zeros((num_agents,), dtype=torch.long)
    class_ids = torch.zeros((num_agents,), dtype=torch.long)
    agent_object_ids: list[str] = []
    agent_is_static: list[bool] = []

    for agent_idx, traffic_object in enumerate(objects):
        agent_object_ids.append(str(traffic_object.track_id))
        agent_is_static.append(bool(traffic_object.is_static))
        track_ids[agent_idx] = _track_id_as_int(traffic_object.track_id, agent_idx)
        class_ids[agent_idx] = _class_id_from_label(traffic_object.label_class)
        lwh[agent_idx] = torch.tensor(
            [
                float(traffic_object.aabb.x),
                float(traffic_object.aabb.y),
                float(traffic_object.aabb.z),
            ],
            dtype=torch.float32,
        )

        time_range = traffic_object.trajectory.time_range_us
        for step_idx, timestamp_us in enumerate(timestamps_us):
            if int(timestamp_us) < int(time_range.start) or int(timestamp_us) >= int(
                time_range.stop
            ):
                continue
            pose = traffic_object.trajectory.interpolate_pose(int(timestamp_us))
            xyz[agent_idx, step_idx] = torch.as_tensor(pose.vec3, dtype=torch.float32)
            heading[agent_idx, step_idx] = float(pose.yaw())
            valid_mask[agent_idx, step_idx] = True

    return (
        {
            "xyz": xyz,
            "heading": heading,
            "valid_mask": valid_mask,
            "lwh": lwh,
            "track_ids": track_ids,
            "class_ids": class_ids,
            "num_obstacles": num_agents,
        },
        agent_object_ids,
        agent_is_static,
    )


def _track_id_as_int(track_id: str, fallback_idx: int) -> int:
    try:
        return int(track_id)
    except (TypeError, ValueError):
        return fallback_idx


def _class_id_from_label(label: str) -> int:
    normalized = str(label).lower()
    class_name_to_id = OBSTACLE_CLASS_NAME_TO_ID
    if normalized in class_name_to_id:
        return class_name_to_id[normalized]
    if normalized in {"automobile", "vehicle", "car"}:
        return class_name_to_id["car"]
    if normalized in {"person", "pedestrian"} or normalized.startswith("person"):
        return class_name_to_id["pedestrian"]
    if normalized in {"bicycle", "bike", "cyclist", "cycle"} or normalized.startswith(
        "cycle"
    ):
        return class_name_to_id["cyclist"]
    if normalized in {"truck", "bus", "trailer", "other_vehicle"}:
        return class_name_to_id["truck"]
    return class_name_to_id["others"]


def _build_empty_agents_for_steps(steps: int) -> dict:
    return {
        "xyz": torch.zeros((0, steps, 3), dtype=torch.float32),
        "heading": torch.zeros((0, steps), dtype=torch.float32),
        "valid_mask": torch.zeros((0, steps), dtype=torch.bool),
        "lwh": torch.zeros((0, 3), dtype=torch.float32),
        "track_ids": torch.zeros((0,), dtype=torch.long),
        "class_ids": torch.zeros((0,), dtype=torch.long),
        "num_obstacles": 0,
    }


_CATK_MAP_ELEMENT_NAME_TO_ENVDATA_MAP_KEY = {
    v: k for k, v in ENVDATA_MAP_KEY_TO_CATK_MAP_ELEMENT_NAME.items()
}
_LANE_TYPE_IDS = {
    MAP_ELEMENT_NAME2_TYPEID["lane_lines"],
    MAP_ELEMENT_NAME2_TYPEID["lane_centers"],
    MAP_ELEMENT_NAME2_TYPEID["lane_boundaries"],
}
_ROAD_BOUNDARY_TYPE_ID = MAP_ELEMENT_NAME2_TYPEID["road_boundaries"]


def _validate_map_config(
    *,
    map_element_names: list[str] | None,
    map_polyline_filter_mode: str,
    map_polyline_number_control_mode: str,
) -> None:
    if map_element_names is not None:
        invalid_names = [
            name for name in map_element_names if name not in MAP_ELEMENT_NAME2_TYPEID
        ]
        if invalid_names:
            raise ValueError(f"Invalid map_element_names: {invalid_names}")
    if map_polyline_filter_mode not in {"v_to_ego", "v_to_ego_and_obs", "disabled"}:
        raise ValueError(
            f"Invalid map_polyline_filter_mode: {map_polyline_filter_mode!r}"
        )
    if map_polyline_number_control_mode not in {"adv", "disabled"}:
        raise ValueError(
            "Invalid map_polyline_number_control_mode: "
            f"{map_polyline_number_control_mode!r}"
        )


def preprocess_runtime_map(
    map_data: dict,
    *,
    ego_xyz: torch.Tensor,
    agents_xyz: torch.Tensor,
    agents_valid_mask: torch.Tensor,
    map_element_names: list[str] | None,
    map_polyline_filter_mode: str,
    map_max_pts_to_ego_distance: float,
    map_polyline_number_control_mode: str,
    map_adv_max_lane_polylines_num: int,
    map_adv_max_road_boundary_num: int,
    map_adv_max_other_polylines_num: int,
) -> dict:
    _validate_map_config(
        map_element_names=map_element_names,
        map_polyline_filter_mode=map_polyline_filter_mode,
        map_polyline_number_control_mode=map_polyline_number_control_mode,
    )
    _apply_map_element_selection(map_data, map_element_names)

    if map_polyline_filter_mode != "disabled":
        _apply_map_distance_filter(
            map_data,
            ego_xyz=ego_xyz,
            agents_xyz=agents_xyz,
            agents_valid_mask=agents_valid_mask,
            mode=map_polyline_filter_mode,
            dist_th=map_max_pts_to_ego_distance,
        )

    if map_polyline_number_control_mode == "adv":
        _apply_map_adv_count_filter(
            map_data,
            max_lane_polylines_num=map_adv_max_lane_polylines_num,
            max_road_boundary_num=map_adv_max_road_boundary_num,
            max_other_polylines_num=map_adv_max_other_polylines_num,
        )

    return map_data


def _apply_map_element_selection(
    map_data: dict, map_element_names: list[str] | None
) -> None:
    if map_element_names is None:
        return
    selected_env_keys = {
        _CATK_MAP_ELEMENT_NAME_TO_ENVDATA_MAP_KEY[name]
        for name in map_element_names
        if name in _CATK_MAP_ELEMENT_NAME_TO_ENVDATA_MAP_KEY
    }
    for env_key in ENVDATA_MAP_KEY_TO_CATK_MAP_ELEMENT_NAME:
        if env_key not in selected_env_keys:
            map_data[env_key] = None


def _apply_map_distance_filter(
    map_data: dict,
    *,
    ego_xyz: torch.Tensor,
    agents_xyz: torch.Tensor,
    agents_valid_mask: torch.Tensor,
    mode: str,
    dist_th: float,
) -> None:
    if dist_th <= 0:
        return
    ref_xy = ego_xyz[:, :2]
    if mode == "v_to_ego_and_obs":
        valid_agent_xy = agents_xyz[agents_valid_mask][:, :2]
        if valid_agent_xy.numel() > 0:
            ref_xy = torch.cat([ref_xy, valid_agent_xy], dim=0)

    for element in map_data.values():
        polylines = _get_polylines(element)
        if polylines is None:
            continue
        keep_mask = _polyline_distance_mask(polylines, ref_xy=ref_xy, dist_th=dist_th)
        _apply_layer_mask(element, keep_mask)


def _polyline_distance_mask(
    polylines: torch.Tensor, *, ref_xy: torch.Tensor, dist_th: float
) -> torch.Tensor:
    if polylines.shape[0] == 0:
        return torch.zeros((0,), dtype=torch.bool, device=polylines.device)
    ref_xy = ref_xy.to(device=polylines.device, dtype=polylines.dtype)
    if ref_xy.numel() == 0:
        return torch.zeros(
            (polylines.shape[0],), dtype=torch.bool, device=polylines.device
        )
    diff = polylines[:, :, None, :2] - ref_xy[None, None, :, :]
    min_dist_sq = diff.square().sum(dim=-1).amin(dim=(1, 2))
    return min_dist_sq <= dist_th**2


def _apply_map_adv_count_filter(
    map_data: dict,
    *,
    max_lane_polylines_num: int,
    max_road_boundary_num: int,
    max_other_polylines_num: int,
) -> None:
    layer_entries = []
    labels = []
    for element in map_data.values():
        polylines = _get_polylines(element)
        if polylines is None:
            continue
        label = element["label"]
        layer_entries.append((element, polylines.shape[0]))
        labels.append(label.to(device=polylines.device, dtype=torch.long))

    if not labels:
        return

    all_labels = torch.cat(labels, dim=0)
    keep_mask = _adv_count_mask(
        all_labels,
        max_lane_polylines_num=max_lane_polylines_num,
        max_road_boundary_num=max_road_boundary_num,
        max_other_polylines_num=max_other_polylines_num,
    )

    offset = 0
    for element, count in layer_entries:
        layer_mask = keep_mask[offset : offset + count]
        offset += count
        _apply_layer_mask(element, layer_mask)


def _adv_count_mask(
    labels: torch.Tensor,
    *,
    max_lane_polylines_num: int,
    max_road_boundary_num: int,
    max_other_polylines_num: int,
) -> torch.Tensor:
    lane_mask = torch.zeros_like(labels, dtype=torch.bool)
    for type_id in _LANE_TYPE_IDS:
        lane_mask = lane_mask | (labels == type_id)
    road_boundary_mask = labels == _ROAD_BOUNDARY_TYPE_ID
    other_mask = ~lane_mask & ~road_boundary_mask

    keep_mask = torch.zeros_like(labels, dtype=torch.bool)
    _select_up_to(keep_mask, lane_mask, max_lane_polylines_num)
    _select_up_to(keep_mask, road_boundary_mask, max_road_boundary_num)
    _select_up_to(keep_mask, other_mask, max_other_polylines_num)
    return keep_mask


def _select_up_to(
    target_mask: torch.Tensor, candidate_mask: torch.Tensor, max_count: int
) -> None:
    candidate_indices = torch.where(candidate_mask)[0]
    if candidate_indices.numel() <= max_count:
        target_mask[candidate_indices] = True
        return
    perm = torch.randperm(candidate_indices.shape[0], device=candidate_indices.device)
    target_mask[candidate_indices[perm[:max_count]]] = True


def _get_polylines(element: dict | None) -> torch.Tensor | None:
    if not isinstance(element, dict):
        return None
    polylines = element.get("polylines")
    if not torch.is_tensor(polylines) or polylines.shape[0] == 0:
        return None
    return polylines


def _apply_layer_mask(element: dict, keep_mask: torch.Tensor) -> None:
    keep_mask = keep_mask.to(device=element["polylines"].device, dtype=torch.bool)
    if keep_mask.sum().item() == 0:
        element["polylines"] = element["polylines"][:0]
        element["label"] = element["label"][:0]
        return
    element["polylines"] = element["polylines"][keep_mask]
    element["label"] = element["label"][keep_mask]
    for optional_key in ("polylines_styles", "polylines_colors"):
        value = element.get(optional_key)
        if torch.is_tensor(value) and value.shape[0] == keep_mask.shape[0]:
            element[optional_key] = value[keep_mask]
