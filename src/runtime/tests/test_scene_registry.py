# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from types import SimpleNamespace

import pytest
from alpasim_runtime.config import (
    ClipDataSourceConfig,
    SceneConfig,
    SimulationConfig,
    UserSimulatorConfig,
)
from alpasim_runtime.data_source_loader import DataSourceInfo
from alpasim_runtime.scene_registry import SceneRegistry
from alpasim_utils.types import T0_AUTO, ClipDataSource, Scene, parse_scene_key
from omegaconf import OmegaConf


def _clip_info(
    clip_id: str = "clipgt-a",
    uuid: str = "uuid-a",
    duration_us: int = 10_000_000,
) -> DataSourceInfo:
    metadata = SimpleNamespace(
        uuid=uuid,
        time_range=SimpleNamespace(start=100_000_000, end=100_000_000 + duration_us),
    )
    return DataSourceInfo(
        source=ClipDataSource(clip_id=clip_id, usdz_uuid=uuid),
        metadata=metadata,
    )


def _simulation_config(force_gt_duration_us: int = 1_600_000) -> SimulationConfig:
    return SimulationConfig(
        n_sim_steps=20,
        n_rollouts=1,
        control_timestep_us=100_000,
        force_gt_duration_us=force_gt_duration_us,
    )


def _explicit_scene(policy_start_us: int = 3_000_000) -> SceneConfig:
    return SceneConfig(
        source=ClipDataSourceConfig(clip_id="clipgt-a", usdz_uuid="uuid-a"),
        policy_start_us=policy_start_us,
    )


def test_user_config_schema_has_one_complete_scene_projection() -> None:
    schema = OmegaConf.structured(UserSimulatorConfig)
    scene_schema = OmegaConf.structured(SceneConfig)

    assert OmegaConf.get_type(schema) is UserSimulatorConfig
    assert list(scene_schema.keys()) == ["source", "policy_start_us", "n_rollouts"]
    assert OmegaConf.is_missing(scene_schema, "source")
    assert OmegaConf.is_missing(scene_schema, "policy_start_us")


def test_scene_config_round_trips_nested_data_source() -> None:
    config = OmegaConf.merge(
        OmegaConf.structured(SceneConfig),
        {
            "source": {"clip_id": "clipgt-a", "usdz_uuid": "uuid-a"},
            "policy_start_us": 3_000_000,
        },
    )

    scene = OmegaConf.to_object(config)

    assert isinstance(scene, SceneConfig)
    assert scene.source.clip_id == "clipgt-a"
    assert scene.source.usdz_uuid == "uuid-a"


def test_explicit_scene_preserves_policy_start() -> None:
    registry = SceneRegistry([_explicit_scene()], [_clip_info()], _simulation_config())

    resolved = registry.resolve(Scene(ClipDataSource("clipgt-a", "uuid-a"), 3_000_000))

    assert resolved.scene_key == "clipgt-a@t0=3000000"
    assert resolved.source == ClipDataSource("clipgt-a", "uuid-a")


def test_fork_resolves_request_time_policy_start() -> None:
    registry = SceneRegistry(
        [
            SceneConfig(
                source=ClipDataSourceConfig(clip_id="clipgt-a", usdz_uuid="uuid-a"),
                policy_start_us=T0_AUTO,
            )
        ],
        [_clip_info()],
        _simulation_config(),
    )

    resolved = registry.resolve_fork(
        Scene(ClipDataSource("clipgt-a", "uuid-a"), 3_000_000)
    )

    assert resolved.scene_key == "clipgt-a@t0=3000000"


def _fork_registry(duration_us: int) -> SceneRegistry:
    return SceneRegistry(
        [
            SceneConfig(
                source=ClipDataSourceConfig(clip_id="clipgt-a", usdz_uuid="uuid-a"),
                policy_start_us=T0_AUTO,
            )
        ],
        [_clip_info(duration_us=duration_us)],
        _simulation_config(),
    )


def test_fork_near_clip_end_does_not_require_the_full_horizon() -> None:
    # The configured horizon from policy start 3.0 s ends at 3.4 s.
    registry = _fork_registry(duration_us=3_100_000)

    resolved = registry.resolve_fork(
        Scene(ClipDataSource("clipgt-a", "uuid-a"), 3_000_000)
    )

    assert resolved.scene_key == "clipgt-a@t0=3000000"


def test_fork_requires_a_control_step_before_clip_end() -> None:
    registry = _fork_registry(duration_us=3_099_999)

    with pytest.raises(ValueError, match="no control step fits"):
        registry.resolve_fork(Scene(ClipDataSource("clipgt-a", "uuid-a"), 3_000_000))


def test_force_gt_validation_does_not_change_scene_address() -> None:
    scene_config = _explicit_scene()
    short_history = SceneRegistry(
        [scene_config], [_clip_info()], _simulation_config(1_000_000)
    )
    long_history = SceneRegistry(
        [scene_config], [_clip_info()], _simulation_config(2_000_000)
    )

    requested = Scene(ClipDataSource("clipgt-a", "uuid-a"), 3_000_000)
    assert short_history.resolve(requested).scene_key == "clipgt-a@t0=3000000"
    assert long_history.resolve(requested).scene_key == "clipgt-a@t0=3000000"


def test_auto_is_distinct_from_explicit_zero() -> None:
    registry = SceneRegistry(
        [
            SceneConfig(
                source=ClipDataSourceConfig(clip_id="clipgt-a", usdz_uuid="uuid-a"),
                policy_start_us=T0_AUTO,
            ),
            SceneConfig(
                source=ClipDataSourceConfig(clip_id="clipgt-a", usdz_uuid="uuid-a"),
                policy_start_us=0,
            ),
        ],
        [_clip_info()],
        _simulation_config(force_gt_duration_us=0),
    )

    assert (
        registry.resolve(Scene(ClipDataSource("clipgt-a", "uuid-a"), T0_AUTO)).scene_key
        == "clipgt-a@t0=AUTO"
    )
    assert (
        registry.resolve(Scene(ClipDataSource("clipgt-a", "uuid-a"), 0)).scene_key
        == "clipgt-a@t0=0"
    )


def test_unknown_scene_address_fails_lookup() -> None:
    registry = SceneRegistry([_explicit_scene()], [_clip_info()], _simulation_config())

    with pytest.raises(KeyError):
        registry.resolve(Scene(ClipDataSource("clipgt-a", "uuid-a"), T0_AUTO))


def test_request_descriptor_must_match_configured_scene() -> None:
    registry = SceneRegistry([_explicit_scene()], [_clip_info()], _simulation_config())

    with pytest.raises(ValueError, match="configured descriptor"):
        registry.resolve(Scene(ClipDataSource("clipgt-a", "other-uuid"), 3_000_000))


def test_duplicate_scene_key_is_rejected() -> None:
    scene = _explicit_scene()
    with pytest.raises(ValueError, match="Duplicate scene key"):
        SceneRegistry([scene, scene], [_clip_info()], _simulation_config())


def test_explicit_scene_requires_force_gt_history() -> None:
    with pytest.raises(ValueError, match="does not provide force_gt_duration_us"):
        SceneRegistry(
            [_explicit_scene(policy_start_us=1_000_000)],
            [_clip_info()],
            _simulation_config(),
        )


def test_explicit_scene_requires_full_horizon() -> None:
    with pytest.raises(ValueError, match="full horizon ends"):
        SceneRegistry(
            [_explicit_scene()],
            [_clip_info(duration_us=3_399_999)],
            _simulation_config(),
        )


def test_explicit_scene_accepts_horizon_ending_at_clip_boundary() -> None:
    registry = SceneRegistry(
        [_explicit_scene()],
        [_clip_info(duration_us=3_400_000)],
        _simulation_config(),
    )

    assert registry.scenes[0].scene_key == "clipgt-a@t0=3000000"


def test_unhandled_timing_rejection_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "alpasim_runtime.scene_registry.evaluate_scene_timing",
        lambda **_: SimpleNamespace(rejection_reason="future_reason"),
    )

    with pytest.raises(ValueError, match="Unhandled scene timing rejection reason"):
        SceneRegistry([_explicit_scene()], [_clip_info()], _simulation_config())


def test_scene_descriptor_must_match_loaded_data_source() -> None:
    with pytest.raises(ValueError, match="loaded descriptor"):
        SceneRegistry(
            [_explicit_scene()],
            [_clip_info(uuid="other-uuid")],
            _simulation_config(),
        )


@pytest.mark.parametrize(
    "scene_key",
    [
        "clipgt-a",
        "@t0=AUTO",
        "clipgt-a@t0=",
        "clipgt-a@t0=auto",
        "clipgt-a@t0=+1",
        "clipgt-a@t0=-1",
        "clipgt-a@t0=01",
        "clipgt-a@t0=١",
    ],
)
def test_parse_scene_key_rejects_noncanonical_values(scene_key: str) -> None:
    with pytest.raises(ValueError):
        parse_scene_key(scene_key)


@pytest.mark.parametrize(
    ("scene_key", "address"),
    [
        ("clipgt-a@t0=AUTO", ("clipgt-a", T0_AUTO)),
        ("clipgt-a@t0=0", ("clipgt-a", 0)),
        ("clipgt-a@t0=42", ("clipgt-a", 42)),
    ],
)
def test_parse_scene_key_round_trips_boundary_values(
    scene_key: str, address: tuple[str, int | str]
) -> None:
    assert parse_scene_key(scene_key) == address
