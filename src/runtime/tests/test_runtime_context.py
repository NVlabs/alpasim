# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from alpasim_runtime.config import UserSimulatorConfig
from alpasim_runtime.runtime_context import build_runtime_context
from omegaconf import OmegaConf


def _make_user_config_dictconfig():
    return OmegaConf.merge(
        OmegaConf.structured(UserSimulatorConfig),
        {
            "simulation_config": {
                "n_sim_steps": 1,
                "n_rollouts": 1,
            },
            "scenes": [{"scene_id": "clipgt-a"}],
            "endpoints": {
                "driver": {"n_concurrent_rollouts": 1},
                "sensorsim": {"n_concurrent_rollouts": 1},
                "physics": {"n_concurrent_rollouts": 1},
                "trafficsim": {"n_concurrent_rollouts": 1},
                "controller": {"n_concurrent_rollouts": 1},
            },
            "nr_workers": 1,
        },
    )


@pytest.mark.asyncio
async def test_build_runtime_context_skips_scene_validation_without_dataclass_replace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_config = _make_user_config_dictconfig()
    network_config = SimpleNamespace()
    version_ids = SimpleNamespace()
    eval_config = SimpleNamespace()
    validate_scenarios = AsyncMock()

    monkeypatch.setattr(
        "alpasim_runtime.runtime_context.parse_simulator_config",
        lambda *_args, **_kwargs: SimpleNamespace(
            user=user_config,
            network=network_config,
        ),
    )
    monkeypatch.setattr(
        "alpasim_runtime.runtime_context.typed_parse_config",
        lambda *_args, **_kwargs: eval_config,
    )
    monkeypatch.setattr(
        "alpasim_runtime.runtime_context.gather_versions_from_addresses",
        AsyncMock(return_value=version_ids),
    )
    monkeypatch.setattr(
        "alpasim_runtime.runtime_context.validate_scenarios",
        validate_scenarios,
    )
    monkeypatch.setattr(
        "alpasim_runtime.runtime_context.Artifact.discover_from_glob",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "alpasim_runtime.runtime_context.create_address_pools",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "alpasim_runtime.runtime_context.compute_max_in_flight",
        lambda *_args, **_kwargs: 1,
    )

    context = await build_runtime_context(
        user_config_path="u.yaml",
        network_config_path="n.yaml",
        eval_config_path="e.yaml",
        usdz_glob="/tmp/*.usdz",
        validate_config_scenes=False,
    )

    validate_scenarios.assert_awaited_once()
    validated_config = validate_scenarios.await_args.args[0]
    assert validated_config.user.scenes == []
    assert context.config.user.scenes[0].scene_id == "clipgt-a"
