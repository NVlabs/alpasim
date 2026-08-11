# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from types import SimpleNamespace

from alpasim_runtime.config import SceneConfig, SimulationConfig
from alpasim_runtime.simulate.__main__ import build_simulation_request


def _user_config(scenes: list[SceneConfig], global_offset_us: int) -> SimpleNamespace:
    # build_simulation_request only reads these three attributes.
    return SimpleNamespace(
        scenes=scenes,
        simulation_config=SimulationConfig(
            n_sim_steps=1, n_rollouts=3, start_time_offset_us=global_offset_us
        ),
        enable_autoresume=False,
    )


def test_build_simulation_request_resolves_start_time_offset(tmp_path) -> None:
    """The per-scene offset overrides the global default, else inherits it."""
    user_config = _user_config(
        scenes=[
            SceneConfig(scene_id="scene-a"),  # inherits the global offset
            SceneConfig(scene_id="scene-b", start_time_offset_us=2_000_000),  # override
        ],
        global_offset_us=1_000_000,
    )

    request = build_simulation_request(user_config, str(tmp_path))

    specs = {spec.scenario_id: spec for spec in request.rollout_specs}
    assert specs["scene-a"].start_time_offset_us == 1_000_000
    assert specs["scene-b"].start_time_offset_us == 2_000_000
    # Rollout counts still resolve the same way (unchanged behavior).
    assert specs["scene-a"].nr_rollouts == 3
    assert specs["scene-b"].nr_rollouts == 3
