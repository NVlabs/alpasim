# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from alpasim_utils.trajdata_data_source import TrajdataDataSource

from trajdata.maps.vec_map_elements import MapElementType


class _States(SimpleNamespace):
    def __len__(self) -> int:
        return len(self.position3d)


class _Trajectory(SimpleNamespace):
    def __len__(self) -> int:
        return len(self.positions)


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
            env_name="test-env",
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


def test_map_coordinate_transform_includes_road_area_rings() -> None:
    scene_cache = MagicMock()
    source = TrajdataDataSource(
        scene=SimpleNamespace(name="test-scene"),
        scene_cache=scene_cache,
    )
    source._rig = SimpleNamespace(
        world_to_nre=np.array(
            [
                [1.0, 0.0, 0.0, -100.0],
                [0.0, 1.0, 0.0, -200.0],
                [0.0, 0.0, 1.0, -300.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        trajectory=_Trajectory(positions=np.array([[0.0, 0.0, 3.0]])),
    )
    exterior = SimpleNamespace(
        points=np.array([[101.0, 202.0, 4.0], [103.0, 204.0, 5.0]])
    )
    interior_hole = SimpleNamespace(
        points=np.array([[101.5, 202.5, 4.5], [102.5, 203.5, 4.5]])
    )
    road_area = SimpleNamespace(
        exterior_polygon=exterior,
        interior_holes=[interior_hole],
    )
    vec_map = SimpleNamespace(
        lanes=None,
        elements={MapElementType.ROAD_AREA: {"area": road_area}},
    )

    source._apply_coordinate_transform_to_map(vec_map)

    np.testing.assert_allclose(
        exterior.points,
        [[1.0, 2.0, 7.0], [3.0, 4.0, 8.0]],
    )
    np.testing.assert_allclose(
        interior_hole.points,
        [[1.5, 2.5, 7.5], [2.5, 3.5, 7.5]],
    )
