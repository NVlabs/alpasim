# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from decimal import Decimal
from math import ceil
from typing import List, Optional, Set, Tuple

from trajdata.data_structures.agent import AgentMetadata, AgentType


def agent_types(
    agents: List[AgentMetadata], no_types: Set[AgentType], only_types: Set[AgentType]
) -> List[AgentMetadata]:
    agents_list: List[AgentMetadata] = agents

    if no_types is not None:
        agents_list = [agent for agent in agents_list if agent.type not in no_types]

    if only_types is not None:
        agents_list = [agent for agent in agents_list if agent.type in only_types]

    return agents_list


def all_agents_excluded_types(
    no_types: Optional[List[AgentType]], agents: List[AgentMetadata]
) -> bool:
    return no_types is not None and all(
        agent_info.type in no_types for agent_info in agents
    )


def no_agent_included_types(
    only_types: Optional[List[AgentType]], agents: List[AgentMetadata]
) -> bool:
    return only_types is not None and all(
        agent_info.type not in only_types for agent_info in agents
    )


def get_valid_ts(
    agent_info: AgentMetadata,
    dt: float,
    history_sec: Tuple[Optional[float], Optional[float]],
    future_sec: Tuple[Optional[float], Optional[float]],
) -> Tuple[int, int]:
    """The returned timesteps are both inclusive.

    Args:
        agent_info (AgentMetadata): _description_
        dt (float): _description_
        history_sec (Tuple[Optional[float], Optional[float]]): _description_
        future_sec (Tuple[Optional[float], Optional[float]]): _description_

    Returns:
        Tuple[int, int]: _description_
    """
    first_valid_ts = agent_info.first_timestep
    if history_sec[0] is not None:
        min_history = ceil(Decimal(str(history_sec[0])) / Decimal(str(dt)))
        first_valid_ts += min_history

    last_valid_ts = agent_info.last_timestep
    if future_sec[0] is not None:
        min_future = ceil(Decimal(str(future_sec[0])) / Decimal(str(dt)))
        last_valid_ts -= min_future

    return first_valid_ts, last_valid_ts


def satisfies_history(
    agent_info: AgentMetadata,
    ts: int,
    dt: float,
    history_sec: Tuple[Optional[float], Optional[float]],
) -> bool:
    if history_sec[0] is not None:
        min_history = ceil(Decimal(str(history_sec[0])) / Decimal(str(dt)))
        agent_history_satisfies = ts - agent_info.first_timestep >= min_history
    else:
        agent_history_satisfies = True

    return agent_history_satisfies


def satisfies_future(
    agent_info: AgentMetadata,
    ts: int,
    dt: float,
    future_sec: Tuple[Optional[float], Optional[float]],
) -> bool:
    if future_sec[0] is not None:
        min_future = ceil(Decimal(str(future_sec[0])) / Decimal(str(dt)))
        agent_future_satisfies = agent_info.last_timestep - ts >= min_future
    else:
        agent_future_satisfies = True

    return agent_future_satisfies


def satisfies_times(
    agent_info: AgentMetadata,
    ts: int,
    dt: float,
    history_sec: Tuple[Optional[float], Optional[float]],
    future_sec: Tuple[Optional[float], Optional[float]],
) -> bool:
    agent_history_satisfies = satisfies_history(agent_info, ts, dt, history_sec)
    agent_future_satisfies = satisfies_future(agent_info, ts, dt, future_sec)
    return agent_history_satisfies and agent_future_satisfies


def no_agent_satisfies_time(
    ts: int,
    dt: float,
    history_sec: Tuple[Optional[float], Optional[float]],
    future_sec: Tuple[Optional[float], Optional[float]],
    agents: List[AgentMetadata],
) -> bool:
    return all(
        not satisfies_times(agent_info, ts, dt, history_sec, future_sec)
        for agent_info in agents
    )
