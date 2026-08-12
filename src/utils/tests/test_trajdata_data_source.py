# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from alpasim_utils.trajdata_data_source import TrajdataDataSource


class _States(SimpleNamespace):
    def __len__(self) -> int:
        return len(self.position3d)


def test_rig_preserves_recorded_linear_dynamics_in_local() -> None:
    ego_agent = SimpleNamespace(
        name="ego",
        last_timestep=2,
        extent=SimpleNamespace(length=5.0, width=2.0, height=1.5),
    )
    states = _States(
        position3d=np.array(
            [[100.0, 200.0, 1.0], [101.0, 201.0, 1.0], [103.0, 202.0, 1.0]]
        ),
        heading=np.array([0.0, 0.1, 0.2]),
        velocity=np.array([[4.0, -1.0], [5.0, -2.0], [6.0, -3.0]]),
        acceleration=np.array([[0.5, -0.1], [0.6, -0.2], [0.7, -0.3]]),
    )
    scene_cache = MagicMock()
    scene_cache.get_agent_history.return_value = (states, None)
    source = TrajdataDataSource(
        scene=SimpleNamespace(
            agents=[ego_agent],
            dt=0.1,
            name="test-scene",
            data_access_info={},
        ),
        scene_cache=scene_cache,
    )

    rig = source.rig

    scene_cache.set_obs_format.assert_called_once_with("x,y,z,xd,yd,xdd,ydd,h")
    np.testing.assert_allclose(
        rig.recorded_rig_linear_velocities_in_local,
        [[4.0, -1.0, 0.0], [5.0, -2.0, 0.0], [6.0, -3.0, 0.0]],
    )
    np.testing.assert_allclose(
        rig.recorded_rig_linear_accelerations_in_local,
        [[0.5, -0.1, 0.0], [0.6, -0.2, 0.0], [0.7, -0.3, 0.0]],
    )
    np.testing.assert_array_equal(
        rig.trajectory.timestamps_us,
        np.array([0, 100_000, 200_000], dtype=np.uint64),
    )
