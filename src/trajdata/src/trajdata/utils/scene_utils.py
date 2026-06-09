# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from math import ceil
from typing import List, Optional, Union

from trajdata.data_structures import AgentMetadata, Scene, SceneMetadata


def enforce_desired_dt(
    scene_info: Union[Scene, SceneMetadata],
    desired_dt: Optional[float],
    dry_run: bool = False,
) -> bool:
    """Enforces that a scene's data is at the desired frequency (specified by desired_dt
    if it's not None) through interpolation.

    Args:
        scene_info (Scene | SceneMetadata): The scene to interpolate to the desired data frequency.
        desired_dt (Optional[float]): The desired data timestep difference (in seconds).
        dry_run (bool): If True, only check if the scene meets the desired data frequency (without modifying scene_info). Defaults to False.

    Returns:
        bool: True if the scene was modified (or would be modified if dry_run=True), False otherwise.
    """
    if desired_dt is not None and scene_info.dt != desired_dt:
        if not dry_run and scene_info.dt > desired_dt:
            interpolate_scene_dt(scene_info, desired_dt)
        elif not dry_run and scene_info.dt < desired_dt:
            subsample_scene_dt(scene_info, desired_dt)
        return True

    return False


def interpolate_scene_dt(scene: Scene, desired_dt: float) -> None:
    dt_ratio: float = scene.dt / desired_dt
    if not dt_ratio.is_integer():
        raise ValueError(
            f"Cannot interpolate scene: {scene.dt} is not integer divisible by {desired_dt} for {str(scene)}"
        )

    dt_factor: int = int(dt_ratio)

    # E.g., the scene is currently at dt = 0.5s (2 Hz),
    # but we want desired_dt = 0.1s (10 Hz).
    scene.length_timesteps = (scene.length_timesteps - 1) * dt_factor + 1
    agent_presence: List[List[AgentMetadata]] = [
        [] for _ in range(scene.length_timesteps)
    ]
    for agent in scene.agents:
        agent.first_timestep *= dt_factor
        agent.last_timestep *= dt_factor

        for scene_ts in range(agent.first_timestep, agent.last_timestep + 1):
            agent_presence[scene_ts].append(agent)

    scene.update_agent_info(scene.agents, agent_presence)
    scene.dt = desired_dt
    # Note we do not touch scene_info.env_metadata.dt, this will serve as our
    # source of the "original" data dt information.


def subsample_scene_dt(scene: Scene, desired_dt: float) -> None:
    dt_ratio: float = desired_dt / scene.dt

    if not is_integer_robust(dt_ratio):
        raise ValueError(
            f"Cannot subsample scene: {desired_dt} is not integer divisible by {scene.dt} for {str(scene)}"
        )

    dt_factor: int = int(round(dt_ratio))

    # E.g., the scene is currently at dt = 0.1s (10 Hz),
    # but we want desired_dt = 0.5s (2 Hz).
    scene.length_timesteps = (scene.length_timesteps - 1) // dt_factor + 1
    agent_presence: List[List[AgentMetadata]] = [
        [] for _ in range(scene.length_timesteps)
    ]
    for agent in scene.agents:
        # Need to be careful with the first timestep, since agents can have
        # first timesteps that are not exactly divisible by the dt_factor.
        agent.first_timestep = ceil(agent.first_timestep / dt_factor)
        agent.last_timestep //= dt_factor

        for scene_ts in range(agent.first_timestep, agent.last_timestep + 1):
            agent_presence[scene_ts].append(agent)

    scene.update_agent_info(scene.agents, agent_presence)
    scene.dt = desired_dt
    # Note we do not touch scene_info.env_metadata.dt, this will serve as our
    # source of the "original" data dt information.


def is_integer_robust(x):
    return abs(x - round(x)) < 1e-6
