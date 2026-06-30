# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

import torch
from alpasim_utils.geometry import Trajectory


class EgoEnvData(TypedDict, total=False):
    xyz: torch.Tensor  # (T, 3)
    heading: torch.Tensor  # (T,)
    lwh: torch.Tensor  # (3,)


class AgentEnvData(TypedDict, total=False):
    xyz: torch.Tensor  # (num_agents, T, 3)
    heading: torch.Tensor  # (num_agents, T)
    valid_mask: torch.Tensor  # (num_agents, T)
    lwh: torch.Tensor  # (num_agents, 3)
    track_ids: torch.Tensor
    class_ids: torch.Tensor
    num_obstacles: int


class EnvMetadata(TypedDict, total=False):
    curr_t: int
    sample_start_t_us: int
    agent_object_ids: list[str]
    agent_is_static: list[bool]


class SimEnvData(TypedDict, total=False):
    ego: EgoEnvData
    agents: AgentEnvData
    env: EnvMetadata
    map: dict[str, Any]
    metadata: dict[str, Any]
    current_time_us: torch.Tensor


@dataclass
class SessionState:
    """Mutable per-session traffic state.

    ``closed_loop_trajectories`` is seeded from logged trajectories and then
    mutated as runtime updates and CATK predictions arrive. It is the durable
    history used to resample each CATK input window. ``env_data`` is the current
    resampled CATK working window and may be rebuilt on each request.
    """

    session_uuid: str
    scene_id: str
    current_ts_us: int
    closed_loop_trajectories: dict[str, Trajectory]
    env_data: SimEnvData
    handover_time_us: int
