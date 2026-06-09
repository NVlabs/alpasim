# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from typing import Dict  # Just for type annotations

import numpy as np
from tqdm import trange

from trajdata import AgentBatch, UnifiedDataset
from trajdata.data_structures.scene_metadata import Scene
from trajdata.data_structures.state import StateArray  # Just for type annotations
from trajdata.simulation import SimulationScene

dataset = UnifiedDataset(
    desired_data=["nusc_mini"],
    data_dirs={  # Remember to change this to match your filesystem!
        "nusc_mini": "~/datasets/nuScenes",
    },
)

desired_scene: Scene = dataset.get_scene(scene_idx=0)
sim_scene = SimulationScene(
    env_name="nusc_mini_sim",
    scene_name="sim_scene",
    scene=desired_scene,
    dataset=dataset,
    init_timestep=0,
    freeze_agents=True,
)

obs: AgentBatch = sim_scene.reset()
for t in trange(1, sim_scene.scene.length_timesteps):
    new_xyzh_dict: Dict[str, StateArray] = dict()

    # Everything inside the forloop just sets
    # agents' next states to their current ones.
    for idx, agent_name in enumerate(obs.agent_name):
        curr_yaw = obs.curr_agent_state[idx].heading.item()
        curr_pos = obs.curr_agent_state[idx].position.numpy()

        next_state = np.zeros((4,))
        next_state[:2] = curr_pos
        next_state[-1] = curr_yaw
        new_xyzh_dict[agent_name] = StateArray.from_array(next_state, "x,y,z,h")

    obs = sim_scene.step(new_xyzh_dict)
