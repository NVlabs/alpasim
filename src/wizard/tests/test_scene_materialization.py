# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from alpasim_utils.scene_timing import SceneTimingRejectionReason
from alpasim_utils.types import T0_AUTO, ClipDataSource, Scene
from alpasim_wizard.context import fetch_scenes
from alpasim_wizard.scenes.materialization import materialize_scene_selection
from alpasim_wizard.scenes.sceneset import USDZManager, USDZTimingMetadata


def _scene(clip_id: str, uuid: str, policy_start_us: int | str) -> Scene:
    return Scene(
        ClipDataSource(clip_id=clip_id, usdz_uuid=uuid),
        policy_start_us,  # type: ignore[arg-type]
    )


def _metadata(uuid: str, clip_id: str, duration_us: int) -> USDZTimingMetadata:
    return USDZTimingMetadata(
        uuid=uuid,
        clip_id=clip_id,
        start_us=100_000_000,
        end_us=100_000_000 + duration_us,
    )


def test_materialization_preserves_order_and_skips_metadata_for_history_rejection() -> (
    None
):
    scenes = [
        _scene("clipgt-auto", "uuid-auto", T0_AUTO),
        _scene("clipgt-history", "uuid-history", 1_000_000),
        _scene("clipgt-short", "uuid-short", 14_000_000),
        _scene("clipgt-boundary", "uuid-boundary", 14_000_000),
    ]
    manager = MagicMock()
    manager.load_timing_metadata.return_value = {
        "uuid-short": _metadata("uuid-short", "clipgt-short", 19_733_638),
        "uuid-boundary": _metadata("uuid-boundary", "clipgt-boundary", 20_000_000),
    }

    result = materialize_scene_selection(
        scenes,
        manager,
        force_gt_duration_us=1_600_000,
        n_sim_steps=38,
        control_timestep_us=200_000,
    )

    assert [scene.scene_key for scene in result.accepted_scenes] == [
        "clipgt-auto@t0=AUTO",
        "clipgt-boundary@t0=14000000",
    ]
    assert [rejection.reason for rejection in result.rejections] == [
        SceneTimingRejectionReason.INSUFFICIENT_HISTORY,
        SceneTimingRejectionReason.INSUFFICIENT_HORIZON,
    ]
    assert result.rejection_counts == {
        "insufficient_history": 1,
        "insufficient_horizon": 1,
    }
    requested_sources = manager.load_timing_metadata.call_args.args[0]
    assert [source.usdz_uuid for source in requested_sources] == [
        "uuid-short",
        "uuid-boundary",
    ]


def test_materialization_exposes_conflicting_clip_identities_to_manager(
    tmp_path: Path,
) -> None:
    scenes = [
        _scene("clipgt-a", "uuid-shared", 14_000_000),
        _scene("clipgt-b", "uuid-shared", 14_000_000),
    ]
    manager = USDZManager(clip_catalog=MagicMock(), cache_dir=str(tmp_path))

    with pytest.raises(ValueError, match="maps to both 'clipgt-a' and 'clipgt-b'"):
        materialize_scene_selection(
            scenes,
            manager,
            force_gt_duration_us=1_600_000,
            n_sim_steps=38,
            control_timestep_us=200_000,
        )


def test_fetch_scenes_stages_only_artifacts_with_an_accepted_scene(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    scenes = [
        _scene("clipgt-a", "uuid-a", 13_000_000),
        _scene("clipgt-a", "uuid-a", 14_000_000),
        _scene("clipgt-b", "uuid-b", 14_000_000),
    ]
    manager = MagicMock()
    manager.load_timing_metadata.return_value = {
        "uuid-a": _metadata("uuid-a", "clipgt-a", 19_500_000),
        "uuid-b": _metadata("uuid-b", "clipgt-b", 19_733_638),
    }
    manager.create_sceneset_directory.return_value = str(
        tmp_path / "scenesets/accepted"
    )
    monkeypatch.setattr(
        "alpasim_wizard.context.USDZManager.from_cfg",
        MagicMock(return_value=manager),
    )
    monkeypatch.setattr(
        "alpasim_wizard.context.resolve_scene_groups",
        MagicMock(return_value=(scenes, {})),
    )
    cfg = SimpleNamespace(
        scenes=SimpleNamespace(
            scene_cache=str(tmp_path),
            local_usdz_dir=None,
            sceneset_path=None,
        ),
        runtime=SimpleNamespace(
            simulation_config=SimpleNamespace(
                force_gt_duration_us=1_600_000,
                n_sim_steps=38,
                control_timestep_us=200_000,
            )
        ),
    )

    with caplog.at_level(logging.INFO, logger="alpasim_wizard.context"):
        materialization, scene_groups = fetch_scenes(cfg)

    assert [scene.scene_key for scene in materialization.accepted_scenes] == [
        "clipgt-a@t0=13000000"
    ]
    assert scene_groups == {}
    manager.create_sceneset_directory.assert_called_once_with(["uuid-a"])
    assert cfg.scenes.sceneset_path == "scenesets/accepted"
    assert "requested=3 accepted=1 rejected=2 invalid_pct=66.67%" in caplog.text
    assert "insufficient_history=0 insufficient_horizon=2" in caplog.text


def test_fetch_scenes_mounts_external_symlink_targets_in_scene_services(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenes = [_scene("clipgt-a", "uuid-a", T0_AUTO)]
    manager = MagicMock()
    manager.create_sceneset_directory.return_value = str(
        tmp_path / "scenesets/accepted"
    )
    external_source = tmp_path / "external"
    manager.external_artifact_mounts.return_value = [
        (str(external_source), str(external_source))
    ]
    monkeypatch.setattr(
        "alpasim_wizard.context.USDZManager.from_cfg",
        MagicMock(return_value=manager),
    )
    monkeypatch.setattr(
        "alpasim_wizard.context.resolve_scene_groups",
        MagicMock(return_value=(scenes, {})),
    )
    scene_cache_mount = f"{tmp_path}:/mnt/nre-data"
    cfg = SimpleNamespace(
        scenes=SimpleNamespace(
            scene_cache=str(tmp_path),
            local_usdz_dir=None,
            sceneset_path=None,
        ),
        runtime=SimpleNamespace(
            simulation_config=SimpleNamespace(
                force_gt_duration_us=1_600_000,
                n_sim_steps=38,
                control_timestep_us=200_000,
            )
        ),
        services=SimpleNamespace(
            renderer=SimpleNamespace(volumes=[scene_cache_mount, "/hood:/hood"]),
            physics=SimpleNamespace(volumes=[scene_cache_mount]),
            trafficsim=SimpleNamespace(volumes=[]),
            runtime=SimpleNamespace(volumes=[scene_cache_mount]),
        ),
    )

    fetch_scenes(cfg)

    expected_mount = f"{external_source}:{external_source}:ro"
    assert cfg.scenes.external_artifact_mounts == [expected_mount]
    assert cfg.services.renderer.volumes[-1] == expected_mount
    assert cfg.services.physics.volumes[-1] == expected_mount
    assert cfg.services.runtime.volumes[-1] == expected_mount
    assert cfg.services.trafficsim.volumes == []
    manager.external_artifact_mounts.assert_called_once_with(
        ["uuid-a"],
        "/mnt/nre-data",
    )


@pytest.mark.parametrize(
    ("existing_mount", "error"),
    [
        ("{source}:/mnt/external", "must be read-only"),
        ("{source}:/mnt:ro", "overlaps an existing service mount"),
        ("{source}:/mnt/external/sub:ro", "overlaps an existing service mount"),
    ],
)
def test_fetch_scenes_rejects_unsafe_existing_external_mounts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_mount: str,
    error: str,
) -> None:
    scenes = [_scene("clipgt-a", "uuid-a", T0_AUTO)]
    manager = MagicMock()
    manager.create_sceneset_directory.return_value = str(
        tmp_path / "scenesets/accepted"
    )
    external_source = tmp_path / "external"
    manager.external_artifact_mounts.return_value = [
        (str(external_source), "/mnt/external")
    ]
    monkeypatch.setattr(
        "alpasim_wizard.context.USDZManager.from_cfg",
        MagicMock(return_value=manager),
    )
    monkeypatch.setattr(
        "alpasim_wizard.context.resolve_scene_groups",
        MagicMock(return_value=(scenes, {})),
    )
    scene_cache_mount = f"{tmp_path}:/mnt/nre-data"
    cfg = SimpleNamespace(
        scenes=SimpleNamespace(
            scene_cache=str(tmp_path),
            local_usdz_dir=None,
            sceneset_path=None,
        ),
        runtime=SimpleNamespace(
            simulation_config=SimpleNamespace(
                force_gt_duration_us=1_600_000,
                n_sim_steps=38,
                control_timestep_us=200_000,
            )
        ),
        services=SimpleNamespace(
            renderer=SimpleNamespace(
                volumes=[
                    scene_cache_mount,
                    existing_mount.format(source=external_source),
                ]
            ),
            physics=SimpleNamespace(volumes=[]),
            trafficsim=SimpleNamespace(volumes=[]),
            runtime=SimpleNamespace(volumes=[]),
        ),
    )

    with pytest.raises(ValueError, match=error):
        fetch_scenes(cfg)


def test_fetch_scenes_fails_before_staging_when_all_scenes_are_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenes = [_scene("clipgt-a", "uuid-a", 14_000_000)]
    manager = MagicMock()
    manager.load_timing_metadata.return_value = {
        "uuid-a": _metadata("uuid-a", "clipgt-a", 19_733_638)
    }
    monkeypatch.setattr(
        "alpasim_wizard.context.USDZManager.from_cfg",
        MagicMock(return_value=manager),
    )
    monkeypatch.setattr(
        "alpasim_wizard.context.resolve_scene_groups",
        MagicMock(return_value=(scenes, {})),
    )
    cfg = SimpleNamespace(
        scenes=SimpleNamespace(
            scene_cache=str(tmp_path),
            local_usdz_dir=None,
            sceneset_path=None,
        ),
        runtime=SimpleNamespace(
            simulation_config=SimpleNamespace(
                force_gt_duration_us=1_600_000,
                n_sim_steps=38,
                control_timestep_us=200_000,
            )
        ),
    )

    with pytest.raises(ValueError, match="All selected scenes are invalid"):
        fetch_scenes(cfg)

    manager.create_sceneset_directory.assert_not_called()


def test_fetch_scenes_rejects_a_group_with_no_accepted_scenes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = _scene("clipgt-training", "uuid-training", 13_000_000)
    validation = _scene("clipgt-validation", "uuid-validation", 14_000_000)
    manager = MagicMock()
    manager.load_timing_metadata.return_value = {
        "uuid-training": _metadata("uuid-training", "clipgt-training", 20_000_000),
        "uuid-validation": _metadata(
            "uuid-validation", "clipgt-validation", 19_733_638
        ),
    }
    monkeypatch.setattr(
        "alpasim_wizard.context.USDZManager.from_cfg",
        MagicMock(return_value=manager),
    )
    monkeypatch.setattr(
        "alpasim_wizard.context.resolve_scene_groups",
        MagicMock(
            return_value=(
                [training, validation],
                {"training": [training], "validation": [validation]},
            )
        ),
    )
    cfg = SimpleNamespace(
        scenes=SimpleNamespace(
            scene_cache=str(tmp_path),
            local_usdz_dir=None,
            sceneset_path=None,
        ),
        runtime=SimpleNamespace(
            simulation_config=SimpleNamespace(
                force_gt_duration_us=1_600_000,
                n_sim_steps=38,
                control_timestep_us=200_000,
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="Scene suite group 'validation' has no valid scenes",
    ):
        fetch_scenes(cfg)

    manager.create_sceneset_directory.assert_not_called()
