# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from types import SimpleNamespace

from alpasim_runtime.config import ClipDataSourceConfig, SceneConfig, SimulationConfig
from alpasim_runtime.simulate.__main__ import build_simulation_request
from alpasim_utils.types import T0_AUTO


def _user_config(scenes: list[SceneConfig]) -> SimpleNamespace:
    return SimpleNamespace(
        scenes=scenes,
        simulation_config=SimulationConfig(n_sim_steps=1, n_rollouts=3),
        enable_autoresume=False,
    )


def test_build_simulation_request_sends_auto_as_absent_selector(tmp_path) -> None:
    user_config = _user_config(
        [
            SceneConfig(
                source=ClipDataSourceConfig("clipgt-a", "uuid-a"),
                policy_start_us=T0_AUTO,
            ),
            SceneConfig(
                source=ClipDataSourceConfig("clipgt-b", "uuid-b"),
                policy_start_us=T0_AUTO,
            ),
        ]
    )

    request = build_simulation_request(user_config, str(tmp_path))

    specs = {spec.scene.source.clip.clip_id: spec for spec in request.rollout_specs}
    assert not specs["clipgt-a"].scene.HasField("policy_start_us")
    assert not specs["clipgt-b"].scene.HasField("policy_start_us")
    assert specs["clipgt-a"].nr_rollouts == 3
    assert specs["clipgt-b"].nr_rollouts == 3


def test_build_simulation_request_sends_explicit_selector(tmp_path) -> None:
    user_config = _user_config(
        [
            SceneConfig(
                source=ClipDataSourceConfig("clipgt-a", "uuid-a"),
                policy_start_us=3_000_000,
            )
        ]
    )

    spec = build_simulation_request(user_config, str(tmp_path)).rollout_specs[0]

    assert spec.scene.source.clip.clip_id == "clipgt-a"
    assert spec.scene.source.clip.usdz_uuid == "uuid-a"
    assert spec.scene.HasField("policy_start_us")
    assert spec.scene.policy_start_us == 3_000_000


def test_build_simulation_request_preserves_explicit_zero(tmp_path) -> None:
    user_config = _user_config(
        [
            SceneConfig(
                source=ClipDataSourceConfig("clipgt-a", "uuid-a"),
                policy_start_us=0,
            )
        ]
    )

    spec = build_simulation_request(user_config, str(tmp_path)).rollout_specs[0]

    assert spec.scene.HasField("policy_start_us")
    assert spec.scene.policy_start_us == 0
