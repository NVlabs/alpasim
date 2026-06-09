# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from collections import namedtuple
from typing import Any, List, Optional

from trajdata.data_structures.agent import AgentMetadata
from trajdata.data_structures.environment import EnvMetadata

# Holds scene metadata (e.g., name, dt), but without the memory
# footprint of all the actual underlying scene data.
SceneMetadata = namedtuple("SceneMetadata", ["env_name", "name", "dt", "raw_data_idx"])


class Scene:
    """Holds the data for a particular scene."""

    def __init__(
        self,
        env_metadata: EnvMetadata,
        name: str,
        location: str,
        data_split: str,
        length_timesteps: int,
        raw_data_idx: int,
        data_access_info: Any,
        description: Optional[str] = None,
        agents: Optional[List[AgentMetadata]] = None,
        agent_presence: Optional[List[List[AgentMetadata]]] = None,
    ) -> None:
        self.env_metadata = env_metadata
        self.env_name = env_metadata.name
        self.name = name
        self.location = location
        self.data_split = data_split
        self.dt = env_metadata.dt
        self.length_timesteps = length_timesteps
        self.raw_data_idx = raw_data_idx
        self.data_access_info = data_access_info
        self.description = description
        self.agents = agents
        self.agent_presence = agent_presence

    def length_seconds(self) -> float:
        return self.length_timesteps * self.dt

    def __repr__(self) -> str:
        return "/".join([self.env_name, self.name])

    def update_agent_info(
        self,
        new_agents: List[AgentMetadata],
        new_agent_presence: List[List[AgentMetadata]],
    ) -> None:
        self.agents = new_agents
        self.agent_presence = new_agent_presence

    def to_metadata(self) -> SceneMetadata:
        return SceneMetadata(self.env_name, self.name, self.dt, self.raw_data_idx)
