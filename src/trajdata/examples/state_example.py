# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from collections import defaultdict

import numpy as np
from torch.utils.data import DataLoader

from trajdata import AgentBatch, AgentType, UnifiedDataset
from trajdata.data_structures.state import StateArray, StateTensor


def main():
    dataset = UnifiedDataset(
        desired_data=["lyft_sample-mini_val"],
        centric="agent",
        desired_dt=0.1,
        history_sec=(3.2, 3.2),
        future_sec=(4.8, 4.8),
        only_predict=[AgentType.VEHICLE],
        state_format="x,y,z,xd,yd,xdd,ydd,h",
        agent_interaction_distances=defaultdict(lambda: 30.0),
        incl_robot_future=False,
        incl_raster_map=True,
        raster_map_params={
            "px_per_m": 2,
            "map_size_px": 224,
            "offset_frac_xy": (-0.5, 0.0),
        },
        num_workers=0,
        verbose=True,
        data_dirs={  # Remember to change this to match your filesystem!
            "lyft_sample": "~/datasets/lyft_sample/scenes/sample.zarr",
        },
    )

    print(f"# Data Samples: {len(dataset):,}")

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=dataset.get_collate_fn(),
        num_workers=4,
    )

    # batchElement has properties that correspond to agent states
    ego_state = dataset[0].curr_agent_state_np.copy()
    print(ego_state)

    # StateArray types offer easy conversion to whatever format you want your state
    # e.g. we want x,y position and cos/sin heading:
    print(ego_state.as_format("x,y,c,s"))

    # We can also access elements via properties
    print(ego_state.position3d)
    print(ego_state.velocity)

    # We can set elements of states via properties. E.g., let's reset the heading to 0
    ego_state.heading = 0
    print(ego_state)

    # We can request elements that aren't directly stored in the state, e.g. cos/sin heading
    print(ego_state.heading_vector)

    # However, we can't set properties that aren't directly stored in the state tensor
    try:
        ego_state.heading_vector = 0.0
    except AttributeError as e:
        print(e)

    # Finally, StateArrays are just np.ndarrays under the hood, and any normal np operation
    # should convert them to a normal array
    print(ego_state**2)

    # To convert an np.array into a StateArray, we just need to specify what format it is
    # Note that StateArrays can have an arbitrary number of batch elems
    print(StateArray.from_array(np.random.randn(1, 2, 3), "x,y,z"))

    # Analagous to StateArray wrapping np.arrays, the StateTensor class gives the same
    # functionality to torch.Tensors
    batch: AgentBatch = next(iter(dataloader))
    ego_state_t: StateTensor = batch.curr_agent_state

    print(ego_state_t.as_format("x,y,c,s"))
    print(ego_state_t.position3d)
    print(ego_state_t.velocity)
    ego_state_t.heading = 0
    print(ego_state_t)
    print(ego_state_t.heading_vector)

    # Furthermore, we can use the from_numpy() and numpy() methods to convert to and from
    # StateTensors with the same format
    print(ego_state_t.numpy())
    print(StateTensor.from_numpy(ego_state))


if __name__ == "__main__":
    main()
