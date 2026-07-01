# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import logging
import os

import torch
import yaml
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


class BatchDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


AGENT_TYPE_UNKNOWN = -1
AGENT_TYPE_VEHICLE = 0
AGENT_TYPE_PEDESTRIAN = 1
AGENT_TYPE_CYCLIST = 2
AGENT_TYPE_OTHER = 3

POLYGON_TYPES_LANELINES = 0
POLYGON_TYPES_ROAD_BOUNDARIES = 1
POLYGON_TYPES_WAITLINES = 2
POLYGON_TYPES_CROSSWALKS = 3
POLYGON_TYPES_LANELINES_BOUNDARIES = 4


def agent_type_from_class_ids(
    env_data, class_ids: torch.Tensor, device: str
) -> torch.Tensor:
    obstacle_class_name_2_id = env_data["metadata"]["obstacle_class_name_2_id"]
    agent_type = torch.full(
        (class_ids.shape[0],),
        AGENT_TYPE_OTHER,
        dtype=torch.long,
        device=device,
    )
    for class_name in ("car", "truck"):
        if class_name in obstacle_class_name_2_id:
            agent_type[class_ids == obstacle_class_name_2_id[class_name]] = (
                AGENT_TYPE_VEHICLE
            )
    if "pedestrian" in obstacle_class_name_2_id:
        agent_type[class_ids == obstacle_class_name_2_id["pedestrian"]] = (
            AGENT_TYPE_PEDESTRIAN
        )
    if "cyclist" in obstacle_class_name_2_id:
        agent_type[class_ids == obstacle_class_name_2_id["cyclist"]] = (
            AGENT_TYPE_CYCLIST
        )
    if "others" in obstacle_class_name_2_id:
        agent_type[class_ids == obstacle_class_name_2_id["others"]] = AGENT_TYPE_OTHER
    return agent_type


def _polyline_inside_mask(
    polylines: torch.Tensor, center_xyz: torch.Tensor, dist_th_sq: float
) -> torch.Tensor:
    coord_dim = min(polylines.shape[-1], center_xyz.shape[-1])
    center = center_xyz.reshape(1, 1, -1)[..., :coord_dim]
    dist_sq = (polylines[..., :coord_dim] - center).pow(2).sum(dim=-1)
    return dist_sq.lt(dist_th_sq).sum(dim=1) > 0


def _filter_polylines_and_labels(map_element: dict, is_inside: torch.Tensor) -> bool:
    if is_inside.sum() == 0:
        return False

    map_element["polylines"] = map_element["polylines"][is_inside]

    for optional_key in ("label", "polylines_styles", "polylines_colors"):
        value = map_element.get(optional_key)
        if torch.is_tensor(value) and value.shape[:1] == is_inside.shape[:1]:
            map_element[optional_key] = value[is_inside.to(value.device)]

    # another optional key
    polylines_attrs = map_element.get("polylines_attrs")
    if isinstance(polylines_attrs, dict):
        keep = is_inside.detach().cpu().tolist()
        filtered_attrs = {}
        for attr, attr_val in polylines_attrs.items():
            if isinstance(attr_val, list) and len(attr_val) == len(keep):
                filtered_attrs[attr] = [
                    item for item, keep_item in zip(attr_val, keep) if keep_item
                ]
            elif (
                torch.is_tensor(attr_val) and attr_val.shape[:1] == is_inside.shape[:1]
            ):
                filtered_attrs[attr] = attr_val[is_inside.to(attr_val.device)]
            else:
                filtered_attrs[attr] = attr_val
        map_element["polylines_attrs"] = filtered_attrs

    return True


def load_model_config(yaml_path):
    assert os.path.exists(yaml_path), f"Config file not found: {yaml_path}"
    with open(yaml_path, "r") as f:
        content = f.read()
        cfg = OmegaConf.create(yaml.safe_load(content))
    OmegaConf.resolve(cfg)
    return cfg


def filter_map(env_data, center_xyz: torch.Tensor, distance_th: float):
    """
    Filter new EnvData map polylines by ego position.

    The artifact-backed map path uses per-polyline ``label`` values as the
    source of map type information, so this path only requires ``polylines``
    and ``label`` on each map element.
    """
    assert distance_th > 0

    map_data = env_data["map"]
    element_names = list(map_data.keys())
    center_xyz = center_xyz.reshape(1, 3)
    dist_th_sq = distance_th**2

    for e in element_names:
        map_element = map_data[e]

        if not isinstance(map_element, dict):
            continue

        if "polylines" not in map_element:
            continue

        polylines = map_element["polylines"]
        if polylines is None or len(polylines) == 0:
            map_data.pop(e)
            logger.info(f"[filter] map element {e} is completely removed ")
            continue

        is_inside = _polyline_inside_mask(polylines, center_xyz, dist_th_sq)
        succ = _filter_polylines_and_labels(map_element, is_inside)
        if not succ:
            logger.info(f"[filter] map element {e} is completely removed ")
            map_data.pop(e)

    return


def _polyline_labels(
    label: object, n_polyline: int, device: torch.device
) -> torch.Tensor:
    label_tensor = torch.as_tensor(label, dtype=torch.long, device=device).reshape(-1)
    if label_tensor.numel() == 0:
        raise ValueError("CATK map element has empty label")
    if label_tensor.numel() == 1:
        return label_tensor.expand(n_polyline)
    if label_tensor.numel() < n_polyline:
        logger.warning(
            "CATK map element label count %s is smaller than "
            "polyline count %s; using first label for the layer",
            label_tensor.numel(),
            n_polyline,
        )
        return label_tensor[:1].expand(n_polyline)
    if label_tensor.numel() > n_polyline:
        logger.warning(
            "CATK map element label count %s is larger than "
            "polyline count %s; truncating labels",
            label_tensor.numel(),
            n_polyline,
        )
    return label_tensor[:n_polyline]


def extract_map_data(
    env_data,
    device: str,
    downsample_lines: bool,
    disable_sub_plyline_type: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict | None, dict | None]:
    """Convert all EnvData map layers with polylines/label into CATK map tensors."""
    rb_polylines = []
    rb_polylines_batch = []
    polyline_triplets = []
    polyline_types = []
    polygon_types = []
    batch = []

    b = 0
    for map_element in env_data["map"].values():
        if not isinstance(map_element, dict):
            continue

        polylines = map_element.get("polylines")
        label = map_element.get("label")
        if polylines is None or label is None or len(polylines) == 0:
            continue
        if polylines.ndim != 3 or polylines.shape[1] < 3:
            continue

        line_labels = _polyline_labels(label, polylines.shape[0], polylines.device)
        road_boundary_mask = line_labels == POLYGON_TYPES_ROAD_BOUNDARIES
        if road_boundary_mask.any():
            rb_polylines.append(polylines[road_boundary_mask])
            rb_polylines_batch.append(b)

        if downsample_lines:
            if polylines.shape[1] < 11:
                continue
            triplet = polylines.unfold(dimension=1, size=11, step=10).transpose(-1, -2)
            indices = torch.linspace(0, 10, steps=3).long()
            flat_triplet_xy = triplet.flatten(0, 1)[:, indices, :2]
        else:
            triplet = polylines.unfold(dimension=1, size=3, step=1).transpose(-1, -2)
            flat_triplet_xy = triplet.flatten(0, 1)[:, :, :2]

        n_windows = triplet.shape[1]
        triplet_labels = line_labels.repeat_interleave(n_windows)
        valid_mask = ~flat_triplet_xy.isnan().any(dim=-1).any(dim=-1)
        flat_triplet_xy = flat_triplet_xy[valid_mask]
        triplet_labels = triplet_labels[valid_mask.to(triplet_labels.device)]

        if flat_triplet_xy.shape[0] == 0:
            continue

        polyline_triplets.append(flat_triplet_xy)
        n_pl = flat_triplet_xy.shape[0]
        batch.append(torch.full((n_pl,), b, dtype=torch.long, device=device))
        polyline_types.append(triplet_labels.to(device=device, dtype=torch.long))
        polygon_types.append(triplet_labels.to(device=device, dtype=torch.long))

    if len(polyline_triplets) == 0:
        return None, None, None, None

    triplets = torch.cat(polyline_triplets, dim=0).to(dtype=torch.float32)
    triplet_thetas = torch.atan2(
        triplets[:, 1, 1] - triplets[:, 0, 1],
        triplets[:, 1, 0] - triplets[:, 0, 0],
    )

    triplets = triplets.to(device)
    triplet_thetas = triplet_thetas.to(device)
    polyline_types = torch.cat(polyline_types, dim=0).to(device)
    if disable_sub_plyline_type:
        polyline_types.fill_(0)

    polyline_extras = {
        "type": polyline_types,
        "pl_type": torch.cat(polygon_types, dim=0).to(device),
        "light_type": torch.zeros_like(polyline_types),
        "batch": torch.cat(batch, dim=0).to(device),
    }
    rb_data = {"rb_polylines": rb_polylines, "rb_polylines_batch": rb_polylines_batch}
    return triplets, triplet_thetas, polyline_extras, rb_data


def extract_ego_data(env_data, t_beg: int, t_end: int, dt: float, device: str) -> dict:
    """
    Args:
        env_data: dict
            'agents'
            'ego'
        t_beg: int
        t_end: int
        dt: float
        device: str

    Returns:
        data["agent"]: Dict
            "role": [n_agent, 3], bool
            "id": [n_agent], int64
            "type": [n_agent], uint8
            "valid_mask": [n_agent, n_step], bool
            "position": [n_agent, n_step, 3], float32
            "heading": [n_agent, n_step], float32
            "velocity": [n_agent, n_step, 2], float32
            "shape": [n_agent, 3], float32
            "batch": [n_agent], int64
            "num_obstacle"
    """

    target_steps = max(t_end - t_beg, 0)
    ego_xyz = env_data["ego"]["xyz"][t_beg:t_end].clone().to(device)
    ego_heading = env_data["ego"]["heading"][t_beg:t_end].clone().to(device)
    if ego_xyz.shape[0] == 0:
        fallback_xyz = env_data["ego"]["xyz"][-1:].clone().to(device)
        fallback_heading = env_data["ego"]["heading"][-1:].clone().to(device)
    else:
        finite_heading = torch.isfinite(ego_heading)
        if finite_heading.any():
            fallback_idx = int(torch.nonzero(finite_heading, as_tuple=False)[-1].item())
        else:
            fallback_idx = ego_heading.shape[0] - 1
        fallback_xyz = ego_xyz[fallback_idx : fallback_idx + 1].clone()
        fallback_heading = ego_heading[fallback_idx : fallback_idx + 1].clone()
        if not finite_heading.all():
            ego_xyz[~finite_heading] = fallback_xyz
            ego_heading[~finite_heading] = fallback_heading

    if ego_xyz.shape[0] < target_steps:
        pad_steps = target_steps - ego_xyz.shape[0]
        ego_xyz = torch.cat([ego_xyz, fallback_xyz.repeat(pad_steps, 1)], dim=0)
        ego_heading = torch.cat(
            [ego_heading, fallback_heading.repeat(pad_steps)], dim=0
        )

    # 1,T,D
    ego_xyz = ego_xyz.unsqueeze(0)
    # 1,T
    ego_heading = ego_heading.unsqueeze(0)
    # 1,3
    ego_lwh = env_data["ego"]["lwh"].clone().to(device).unsqueeze(0)

    ego_role = torch.ones((1, 3), dtype=torch.bool, device=device)
    ego_type = torch.zeros((1), dtype=torch.long, device=device)
    ego_valid_mask = ~torch.isnan(ego_heading)
    ego_id = torch.zeros((1,), dtype=torch.long, device=device)

    # concate ego and agents
    agent_data = {}
    agent_data["valid_mask"] = ego_valid_mask
    agent_data["role"] = ego_role
    agent_data["type"] = ego_type
    agent_data["id"] = ego_id

    # (n,t,3)
    agent_data["position"] = ego_xyz
    agent_data["heading"] = ego_heading
    agent_data["shape"] = ego_lwh

    if agent_data["position"].shape[1] <= 1:
        velocity = torch.zeros(
            (1, agent_data["position"].shape[1], 2),
            dtype=agent_data["position"].dtype,
            device=device,
        )
    else:
        velocity = (
            agent_data["position"][:, 1:, :2] - agent_data["position"][:, :-1, :2]
        ) / dt
        velocity = torch.cat([velocity, velocity[:, -1:]], dim=1)

    agent_data["velocity"] = velocity
    agent_data["batch"] = torch.zeros((1,), dtype=torch.long, device=device)

    return agent_data


def extract_static_agent_freeze_data(
    env_data,
    *,
    static_mask: torch.Tensor,
    curr_t: int,
    target_steps: int,
    device: str,
) -> dict | None:
    if not bool(static_mask.any().item()):
        return None

    source_device = env_data["agents"]["xyz"].device
    static_indices = torch.where(static_mask.to(device=source_device))[0]
    current_xyz = (
        env_data["agents"]["xyz"][static_indices, curr_t, :].clone().to(device)
    )
    current_heading = (
        env_data["agents"]["heading"][static_indices, curr_t].clone().to(device)
    )
    current_valid = (
        env_data["agents"]["valid_mask"][static_indices, curr_t].clone().to(device)
    )
    n_static = int(static_indices.numel())

    agent_data = {
        "valid_mask": current_valid.unsqueeze(1).expand(-1, target_steps).clone(),
        "position": current_xyz.unsqueeze(1).expand(-1, target_steps, -1).clone(),
        "heading": current_heading.unsqueeze(1).expand(-1, target_steps).clone(),
        "shape": env_data["agents"]["lwh"][static_indices].clone().to(device),
        "id": env_data["agents"]["track_ids"][static_indices].clone().to(device).long(),
        "batch": torch.zeros((n_static,), dtype=torch.long, device=device),
        "velocity": torch.zeros(
            (n_static, target_steps, 2),
            dtype=current_xyz.dtype,
            device=device,
        ),
    }
    agent_data["role"] = torch.zeros((n_static, 3), dtype=torch.bool, device=device)
    agent_data["role"][:, 1] = True
    agent_data["role"][:, 2] = True
    agent_data["type"] = agent_type_from_class_ids(
        env_data,
        env_data["agents"]["class_ids"][static_indices].clone().to(device),
        device,
    )
    return agent_data


def static_agent_freeze_mask(
    env_data, *, n_agent: int, curr_t: int, device: str
) -> torch.Tensor:
    raw_mask = env_data["env"].get("agent_is_static")
    if raw_mask is None or n_agent <= 0:
        return torch.zeros((n_agent,), dtype=torch.bool, device=device)

    static_mask = torch.as_tensor(raw_mask, dtype=torch.bool, device=device).flatten()
    if int(static_mask.numel()) < n_agent:
        padded = torch.zeros((n_agent,), dtype=torch.bool, device=device)
        padded[: int(static_mask.numel())] = static_mask
        static_mask = padded
    else:
        static_mask = static_mask[:n_agent]

    valid_now = env_data["agents"]["valid_mask"][:n_agent, curr_t].to(device=device)
    return static_mask & valid_now


def _concat_agent_data(first: dict | None, second: dict) -> dict:
    if first is None:
        return second
    return {key: torch.cat([first[key], second[key]], dim=0) for key in second}


def build_freeze_agent_data(
    env_data,
    *,
    curr_t: int,
    target_steps: int,
    dt: float,
    device: str,
) -> tuple[BatchDict, torch.Tensor]:
    n_agent = int(env_data["agents"]["num_obstacles"])
    static_mask = static_agent_freeze_mask(
        env_data,
        n_agent=n_agent,
        curr_t=curr_t,
        device=device,
    )
    freeze_mask = torch.zeros((n_agent + 1,), dtype=torch.bool, device=device)
    freeze_mask[:n_agent] = static_mask
    freeze_mask[-1] = True

    static_data = extract_static_agent_freeze_data(
        env_data,
        static_mask=static_mask,
        curr_t=curr_t,
        target_steps=target_steps,
        device=device,
    )
    ego_data = extract_ego_data(
        env_data,
        t_beg=curr_t,
        t_end=curr_t + target_steps,
        dt=dt,
        device=device,
    )
    freeze_data = BatchDict(
        {
            "agent": _concat_agent_data(static_data, ego_data),
            "num_obstacles": torch.tensor(
                (int(freeze_mask.sum().item()),),
                device=device,
                dtype=torch.long,
            ).reshape(1),
            "num_graphs": 1,
        }
    )
    return freeze_data, freeze_mask


def extract_agents_and_ego_data(
    env_data,
    t_beg: int,
    t_end: int,
    dt: float,
    device: str,
    avoid_fragmented_agents: bool = False,
) -> dict:
    """
    Args:
        env_data: dict
            'agents': T=16,16+X,16+X*2, ...
                'xyz': torch.Tensor, # N,T,3
                'heading': torch.Tensor, # N,T
                'valid_mask': torch.Tensor, # N,T,
                'lwh': torch.Tensor, # N,3
                'track_ids': torch.Tensor, # N
                'class_ids': torch.Tensor, # N
                'num_obstacles': torch.Tensor, # 1,
                # 'timestamps': torch.Tensor, # T, (us)
            'ego'
        t_beg: int
        t_end: int
        dt: float
        device: str


    Returns:
        data["agent"]: Dict
            "role": [n_agent, 3], bool
            "id": [n_agent], int64
            "type": [n_agent], uint8
            "valid_mask": [n_agent, n_step], bool
            "position": [n_agent, n_step, 3], float32
            "heading": [n_agent, n_step], float32
            "velocity": [n_agent, n_step, 2], float32
            "shape": [n_agent, 3], float32
            "batch": [n_agent], int64
            "num_obstacle" [n_agent]
    """

    n_agent = env_data["agents"]["num_obstacles"]

    agent_xyz = env_data["agents"]["xyz"][:n_agent, t_beg:t_end].clone().to(device)
    agent_heading = (
        env_data["agents"]["heading"][:n_agent, t_beg:t_end].clone().to(device)
    )
    agent_lwh = env_data["agents"]["lwh"][:n_agent].clone().to(device)

    agent_valid_mask = (
        env_data["agents"]["valid_mask"][:n_agent, t_beg:t_end].clone().to(device)
    )
    if avoid_fragmented_agents:
        agent_valid = agent_valid_mask.sum(dim=1)
        agent_valid_mask[agent_valid < 2] = False
        agent_curr_valid = agent_valid_mask[:, -1]
        agent_valid_mask[~agent_curr_valid] = False

    agent_role = torch.zeros((n_agent, 3), dtype=torch.bool, device=device)
    agent_role[:, 0] = False  # ego_vehicle
    agent_role[:, 1] = True  # interest
    agent_role[:, 2] = True  # predict

    agent_id = env_data["agents"]["track_ids"][:n_agent].clone().to(device)
    agent_id = agent_id.to(torch.long)
    agent_class_ids = env_data["agents"]["class_ids"][:n_agent].clone().to(device)

    agent_type = agent_type_from_class_ids(env_data, agent_class_ids, device)

    # 1,T,D
    ego_xyz = env_data["ego"]["xyz"][t_beg:t_end].clone().to(device).unsqueeze(0)
    # 1,T
    ego_heading = (
        env_data["ego"]["heading"][t_beg:t_end].clone().to(device).unsqueeze(0)
    )
    # 1,3
    ego_lwh = env_data["ego"]["lwh"].clone().to(device).unsqueeze(0)

    ego_role = torch.ones((1, 3), dtype=torch.bool, device=device)
    ego_type = torch.zeros((1), dtype=torch.long, device=device)
    ego_valid_mask = ~torch.isnan(ego_heading)

    ego_id = torch.zeros((1,), dtype=torch.long, device=device)

    # concate ego and agents
    agent_data = {}
    agent_data["valid_mask"] = torch.cat([agent_valid_mask, ego_valid_mask], dim=0)
    agent_data["role"] = torch.cat([agent_role, ego_role], dim=0)
    agent_data["type"] = torch.cat([agent_type, ego_type], dim=0)
    agent_data["id"] = torch.cat([agent_id, ego_id], dim=0)

    # (n,t,3)
    agent_data["position"] = torch.cat([agent_xyz, ego_xyz], dim=0)
    agent_data["heading"] = torch.cat([agent_heading, ego_heading], dim=0)
    agent_data["shape"] = torch.cat([agent_lwh, ego_lwh], dim=0)

    # clean NaN
    agent_data["position"] = torch.nan_to_num(agent_data["position"])
    agent_data["heading"] = torch.nan_to_num(agent_data["heading"])

    velocity = (
        agent_data["position"][:, 1:, :2] - agent_data["position"][:, :-1, :2]
    ) / dt
    velocity = torch.cat([velocity, velocity[:, -1:]], dim=1)

    agent_data["velocity"] = velocity

    agent_data["batch"] = torch.zeros((n_agent + 1,), dtype=torch.long, device=device)

    return agent_data
