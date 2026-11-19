# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import re
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from alpasim_utils.types import T0_AUTO, ClipDataSource, Scene
from alpasim_wizard.scenes.scene_suite import find_scene_suite_csv, resolve_scene_suites
from alpasim_wizard.scenes.sceneset import (
    USDZManager,
    USDZQueryError,
    resolve_scene_groups,
    resolve_scene_selection,
)
from alpasim_wizard.schema import ScenesConfig, SceneSuiteConfig
from alpasim_wizard.setup_omegaconf import _DEFAULT_CLIP_IDS, update_scene_config
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


def _suite(tmp_path: Path, suite_id: str, rows: str) -> SceneSuiteConfig:
    path = tmp_path / f"{suite_id}.csv"
    path.write_text("clip_id,uuid,event_t0_us\n" + rows)
    return SceneSuiteConfig(csv_path=str(path))


def _catalog(*pairs: tuple[str, str]) -> pl.DataFrame:
    return pl.DataFrame(pairs, schema=["clip_id", "uuid"], orient="row")


def test_find_scene_suite_csv_by_id(tmp_path: Path) -> None:
    suite = tmp_path / "nested" / "suite-a.csv"
    suite.parent.mkdir()
    suite.touch()

    assert find_scene_suite_csv("suite-a", [str(tmp_path)]) == suite.absolute()


@pytest.mark.parametrize("suite_id", ["", "../suite-a", "nested/suite-a"])
def test_find_scene_suite_csv_rejects_invalid_id(tmp_path: Path, suite_id: str) -> None:
    with pytest.raises(ValueError, match="Invalid scene-suite ID"):
        find_scene_suite_csv(suite_id, [str(tmp_path)])


def test_find_scene_suite_csv_rejects_missing_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown scene-suite ID"):
        find_scene_suite_csv("missing", [str(tmp_path)])


def test_find_scene_suite_csv_rejects_ambiguous_id(tmp_path: Path) -> None:
    first = tmp_path / "first" / "same.csv"
    second = tmp_path / "second" / "same.csv"
    first.parent.mkdir()
    second.parent.mkdir()
    first.touch()
    second.touch()

    with pytest.raises(ValueError, match="Scene-suite ID 'same' is ambiguous"):
        find_scene_suite_csv("same", [str(tmp_path)])


def test_resolve_scene_suites_preserves_order_and_deduplicates_address(
    tmp_path: Path,
):
    first = _suite(tmp_path, "first", "clip-b,uuid-b,20\nclip-a,uuid-a,AUTO\n")
    second = _suite(tmp_path, "second", "clip-b,uuid-b,30\nclip-b,uuid-b,20\n")

    scenes = resolve_scene_suites(
        [first, second],
        _catalog(("clip-a", "uuid-a"), ("clip-b", "uuid-b")),
    )

    assert scenes == [
        Scene(ClipDataSource("clip-b", "uuid-b"), 20),
        Scene(ClipDataSource("clip-a", "uuid-a"), T0_AUTO),
        Scene(ClipDataSource("clip-b", "uuid-b"), 30),
    ]


def test_suite_groups_preserve_membership_and_deduplicate_runtime_union(
    tmp_path: Path,
) -> None:
    training = _suite(
        tmp_path,
        "training",
        "clip-a,uuid-a,10\nclip-b,uuid-b,20\n",
    )
    validation = _suite(
        tmp_path,
        "validation",
        "clip-b,uuid-b,20\nclip-a,uuid-a,30\n",
    )
    config = ScenesConfig(
        suite_groups={"training": [training], "validation": [validation]},
        scene_cache=str(tmp_path),
        clip_catalog_csvs=[],
        scene_suite_dirs=[],
    )
    manager = USDZManager(
        clip_catalog=_catalog(("clip-a", "uuid-a"), ("clip-b", "uuid-b")),
        cache_dir=str(tmp_path),
    )

    scenes, groups = resolve_scene_groups(config, manager)

    assert [scene.scene_key for scene in scenes] == [
        "clip-a@t0=10",
        "clip-b@t0=20",
        "clip-a@t0=30",
    ]
    assert {
        name: [scene.scene_key for scene in group] for name, group in groups.items()
    } == {
        "training": ["clip-a@t0=10", "clip-b@t0=20"],
        "validation": ["clip-b@t0=20", "clip-a@t0=30"],
    }


def test_resolve_scene_suites_rejects_empty_configuration():
    with pytest.raises(ValueError, match="must not be empty"):
        resolve_scene_suites([], _catalog(("clip-a", "uuid-a")))


@pytest.mark.parametrize(
    ("event_t0_us", "offset", "expected"),
    [("123", 0, 123), ("123", 7, 130), ("123", -23, 100), ("0", 0, 0)],
)
def test_numeric_event_t0_applies_suite_offset(
    tmp_path: Path, event_t0_us: str, offset: int, expected: int
):
    suite = _suite(tmp_path, "numeric", f"clip-a,uuid-a,{event_t0_us}\n")
    suite.t0_offset_us = offset

    [scene] = resolve_scene_suites([suite], _catalog(("clip-a", "uuid-a")))

    assert scene.policy_start_us == expected


def test_auto_ignores_suite_offset(tmp_path: Path):
    suite = _suite(tmp_path, "auto", "clip-a,uuid-a,AUTO\n")
    suite.t0_offset_us = -100

    [scene] = resolve_scene_suites([suite], _catalog(("clip-a", "uuid-a")))

    assert scene.policy_start_us == T0_AUTO


def test_mixed_auto_and_numeric_t0(tmp_path: Path):
    suite = _suite(
        tmp_path,
        "mixed",
        "clip-a,uuid-a,AUTO\nclip-b,uuid-b,123\n",
    )
    suite.t0_offset_us = 7

    scenes = resolve_scene_suites(
        [suite], _catalog(("clip-a", "uuid-a"), ("clip-b", "uuid-b"))
    )

    assert [scene.policy_start_us for scene in scenes] == [T0_AUTO, 130]


def test_resolve_scene_suites_rejects_missing_pin(tmp_path: Path):
    suite = _suite(tmp_path, "missing", "clip-a,uuid-old,AUTO\n")
    with pytest.raises(ValueError, match="Failed to find pinned"):
        resolve_scene_suites([suite], _catalog(("clip-a", "uuid-new")))


def test_resolve_scene_suites_rejects_uuid_conflict(tmp_path: Path):
    first = _suite(tmp_path, "first", "clip-a,uuid-a,AUTO\n")
    second = _suite(tmp_path, "second", "clip-a,uuid-b,AUTO\n")
    with pytest.raises(ValueError, match="pin multiple UUIDs"):
        resolve_scene_suites([first, second], _catalog(("clip-a", "uuid-a")))


def test_resolve_scene_suites_rejects_duplicate_id(tmp_path: Path):
    first = _suite(tmp_path, "same", "clip-a,uuid-a,AUTO\n")
    duplicate = SceneSuiteConfig(csv_path=first.csv_path)
    with pytest.raises(ValueError, match="Duplicate scene_suite_id"):
        resolve_scene_suites([first, duplicate], _catalog(("clip-a", "uuid-a")))


def test_limit_applies_after_ordered_resolution(tmp_path: Path):
    suite = _suite(tmp_path, "ordered", "clip-b,uuid-b,AUTO\nclip-a,uuid-a,AUTO\n")
    config = ScenesConfig(
        suite_configs=[suite],
        limit_to_first_n=1,
        scene_cache=str(tmp_path),
        clip_catalog_csvs=[],
    )
    manager = SimpleNamespace(
        query_by_scene_suites=lambda _configs: [
            Scene(ClipDataSource("clip-b", "uuid-b"), T0_AUTO),
            Scene(ClipDataSource("clip-a", "uuid-a"), T0_AUTO),
        ]
    )
    assert [
        scene.source.source_id for scene in resolve_scene_selection(config, manager)
    ] == ["clip-b"]


@pytest.mark.parametrize("use_local_clips", [False, True])
def test_direct_and_local_limit_use_source_id_order(
    tmp_path: Path, use_local_clips: bool
):
    config = ScenesConfig(
        clip_ids=None if use_local_clips else ["clip-b", "clip-a"],
        local_usdz_dir=str(tmp_path) if use_local_clips else None,
        limit_to_first_n=1,
        scene_cache=str(tmp_path),
        clip_catalog_csvs=[],
    )
    manager = SimpleNamespace(
        clip_catalog=_catalog(("clip-b", "uuid-b"), ("clip-a", "uuid-a")),
        query_by_clip_ids=lambda _clip_ids: [
            Scene(ClipDataSource("clip-b", "uuid-b"), T0_AUTO),
            Scene(ClipDataSource("clip-a", "uuid-a"), T0_AUTO),
        ],
    )

    assert [
        scene.source.source_id for scene in resolve_scene_selection(config, manager)
    ] == ["clip-a"]


def test_exclusions_apply_before_limiting(tmp_path: Path):
    config = ScenesConfig(
        suite_configs=[
            _suite(tmp_path, "ordered", "clip-b,uuid-b,AUTO\nclip-a,uuid-a,AUTO\n")
        ],
        exclude_clip_ids=["clip-a"],
        limit_to_first_n=1,
        scene_cache=str(tmp_path),
        clip_catalog_csvs=[],
    )
    manager = SimpleNamespace(
        query_by_scene_suites=lambda _configs: [
            Scene(ClipDataSource("clip-b", "uuid-b"), T0_AUTO),
            Scene(ClipDataSource("clip-a", "uuid-a"), T0_AUTO),
        ]
    )

    assert [
        scene.source.source_id for scene in resolve_scene_selection(config, manager)
    ] == ["clip-b"]


def test_exclusions_reject_unselected_clip(tmp_path: Path):
    config = ScenesConfig(
        clip_ids=["clip-a"],
        exclude_clip_ids=["clip-typo"],
        scene_cache=str(tmp_path),
        clip_catalog_csvs=[],
    )
    manager = SimpleNamespace(
        query_by_clip_ids=lambda _clip_ids: [
            Scene(ClipDataSource("clip-a", "uuid-a"), T0_AUTO)
        ]
    )

    with pytest.raises(USDZQueryError, match="not selected"):
        resolve_scene_selection(config, manager)


def test_exclusions_reject_empty_workload(tmp_path: Path):
    config = ScenesConfig(
        clip_ids=["clip-a"],
        exclude_clip_ids=["clip-a"],
        scene_cache=str(tmp_path),
        clip_catalog_csvs=[],
    )
    manager = SimpleNamespace(
        query_by_clip_ids=lambda _clip_ids: [
            Scene(ClipDataSource("clip-a", "uuid-a"), T0_AUTO)
        ]
    )

    with pytest.raises(USDZQueryError, match="removed every selected clip"):
        resolve_scene_selection(config, manager)


def _hydra_overrides(monkeypatch: pytest.MonkeyPatch, *overrides: str) -> None:
    monkeypatch.setattr(
        HydraConfig,
        "get",
        lambda: SimpleNamespace(overrides=SimpleNamespace(task=list(overrides))),
    )


def test_scene_suite_config_cli_override_replaces_inherited_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    override = (
        'scenes.suite_configs=[{csv_path:${repo-relative:"data/scenes/scene_suites/'
        'public_2601.csv"},t0_offset_us:-1000000}]'
    )
    config_dir = Path(__file__).parents[1] / "configs"
    monkeypatch.chdir(tmp_path)
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(config_name="base_config.yaml", overrides=[override])

    _hydra_overrides(monkeypatch, override)
    update_scene_config(config)

    assert config.scenes.clip_ids is None
    assert Path(config.scenes.suite_configs[0].csv_path).name == "public_2601.csv"
    assert config.scenes.suite_configs[0].t0_offset_us == -1_000_000


def test_suite_groups_cli_override_replaces_inherited_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    training = tmp_path / "training.csv"
    validation = tmp_path / "validation.csv"
    training.touch()
    validation.touch()
    override = (
        "+scenes.suite_groups={"
        f"training:[{{csv_path:{training}}}],"
        f"validation:[{{csv_path:{validation}}}]"
        "}"
    )
    config_dir = Path(__file__).parents[1] / "configs"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(config_name="base_config.yaml", overrides=[override])

    _hydra_overrides(monkeypatch, override)
    update_scene_config(config)

    assert config.scenes.clip_ids is None
    assert list(config.scenes.suite_groups) == ["training", "validation"]

    config.scenes.limit_to_first_n = 1
    with pytest.raises(ValueError, match="cannot be combined"):
        update_scene_config(config)


def test_suite_ids_cli_override_normalizes_to_zero_offset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    override = "scenes.suite_ids=[public_2601]"
    config_dir = Path(__file__).parents[1] / "configs"
    monkeypatch.chdir(tmp_path)
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(config_name="base_config.yaml", overrides=[override])

    _hydra_overrides(monkeypatch, override)
    update_scene_config(config)

    assert config.scenes.clip_ids is None
    assert config.scenes.suite_ids is None
    assert Path(config.scenes.suite_configs[0].csv_path).name == "public_2601.csv"
    assert config.scenes.suite_configs[0].t0_offset_us == 0


def test_scene_suite_config_override_key_accepts_spaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hydra_overrides(monkeypatch, "+scenes.suite_configs = [{csv_path:suite.csv}]")
    config = SimpleNamespace(
        scenes=SimpleNamespace(
            clip_ids=list(_DEFAULT_CLIP_IDS),
            suite_ids=None,
            suite_configs=[object()],
            scene_suite_dirs=[],
            local_usdz_dir=None,
        )
    )
    update_scene_config(config)
    assert config.scenes.clip_ids is None


def test_suite_ids_override_key_accepts_spaces(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite-a.csv"
    suite.touch()
    _hydra_overrides(monkeypatch, "scenes.suite_ids = [suite-a]")
    config = SimpleNamespace(
        scenes=SimpleNamespace(
            clip_ids=list(_DEFAULT_CLIP_IDS),
            suite_ids=["suite-a"],
            suite_configs=None,
            scene_suite_dirs=[str(tmp_path)],
            local_usdz_dir=None,
        )
    )

    update_scene_config(config)

    assert config.scenes.clip_ids is None
    assert config.scenes.suite_ids is None
    assert config.scenes.suite_configs == [
        SceneSuiteConfig(csv_path=str(suite.absolute()))
    ]


def test_suite_ids_preserve_authored_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for suite_id in ("suite-a", "suite-b"):
        (tmp_path / f"{suite_id}.csv").touch()
    _hydra_overrides(monkeypatch, "scenes.suite_ids=[suite-b,suite-a]")
    config = SimpleNamespace(
        scenes=SimpleNamespace(
            clip_ids=list(_DEFAULT_CLIP_IDS),
            suite_ids=["suite-b", "suite-a"],
            suite_configs=None,
            scene_suite_dirs=[str(tmp_path)],
            local_usdz_dir=None,
        )
    )

    update_scene_config(config)

    assert [
        Path(suite_config.csv_path).stem for suite_config in config.scenes.suite_configs
    ] == ["suite-b", "suite-a"]


def test_suite_ids_match_explicit_zero_offset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    suite = _suite(tmp_path, "suite-a", "clip-a,uuid-a,123\n")
    _hydra_overrides(monkeypatch, "scenes.suite_ids=[suite-a]")
    config = SimpleNamespace(
        scenes=SimpleNamespace(
            clip_ids=list(_DEFAULT_CLIP_IDS),
            suite_ids=["suite-a"],
            suite_configs=None,
            scene_suite_dirs=[str(tmp_path)],
            local_usdz_dir=None,
        )
    )

    update_scene_config(config)

    assert resolve_scene_suites(
        config.scenes.suite_configs, _catalog(("clip-a", "uuid-a"))
    ) == (resolve_scene_suites([suite], _catalog(("clip-a", "uuid-a"))))


def test_suite_ids_reject_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    _hydra_overrides(monkeypatch, "scenes.suite_ids=[]")
    config = SimpleNamespace(
        scenes=SimpleNamespace(
            clip_ids=None,
            suite_ids=[],
            suite_configs=None,
            scene_suite_dirs=[],
            local_usdz_dir=None,
        )
    )

    with pytest.raises(
        ValueError,
        match=re.escape("scenes.suite_ids must not be empty"),
    ):
        update_scene_config(config)


def test_suite_ids_reject_duplicates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _suite(tmp_path, "suite-a", "clip-a,uuid-a,AUTO\n")
    _hydra_overrides(monkeypatch, "scenes.suite_ids=[suite-a,suite-a]")
    config = SimpleNamespace(
        scenes=SimpleNamespace(
            clip_ids=None,
            suite_ids=["suite-a", "suite-a"],
            suite_configs=None,
            scene_suite_dirs=[str(tmp_path)],
            local_usdz_dir=None,
        )
    )
    update_scene_config(config)

    with pytest.raises(ValueError, match="Duplicate scene_suite_id"):
        resolve_scene_suites(
            config.scenes.suite_configs,
            _catalog(("clip-a", "uuid-a")),
        )


@pytest.mark.parametrize(
    ("clip_ids", "suite_configs", "local_usdz_dir", "message"),
    [
        (["clip-a"], None, None, "either scenes.suite_ids or scenes.clip_ids"),
        (
            None,
            [object()],
            None,
            "either scenes.suite_ids or scenes.suite_configs",
        ),
        (
            None,
            None,
            "/tmp/clips",
            "either scenes.suite_ids or scenes.local_usdz_dir",
        ),
    ],
)
def test_suite_ids_reject_other_selectors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clip_ids: list[str] | None,
    suite_configs: list[object] | None,
    local_usdz_dir: str | None,
    message: str,
) -> None:
    (tmp_path / "suite-a.csv").touch()
    overrides = ["scenes.suite_ids=[suite-a]"]
    if clip_ids is not None:
        overrides.append("scenes.clip_ids=[clip-a]")
    _hydra_overrides(monkeypatch, *overrides)
    config = SimpleNamespace(
        scenes=SimpleNamespace(
            clip_ids=clip_ids,
            suite_ids=["suite-a"],
            suite_configs=suite_configs,
            scene_suite_dirs=[str(tmp_path)],
            local_usdz_dir=local_usdz_dir,
        )
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        update_scene_config(config)


@pytest.mark.parametrize(
    ("clip_ids", "local_usdz_dir", "message"),
    [
        (["clip-a"], None, "either scenes.suite_configs or scenes.clip_ids"),
        (
            None,
            "/tmp/clips",
            "either scenes.suite_configs or scenes.local_usdz_dir",
        ),
    ],
)
def test_scene_suite_configs_reject_other_selectors(
    monkeypatch: pytest.MonkeyPatch,
    clip_ids: list[str] | None,
    local_usdz_dir: str | None,
    message: str,
) -> None:
    overrides = ["scenes.suite_configs=[{csv_path:suite.csv}]"]
    if clip_ids is not None:
        overrides.append("scenes.clip_ids=[clip-a]")
    if local_usdz_dir is not None:
        overrides.append("scenes.local_usdz_dir=/tmp/clips")
    _hydra_overrides(monkeypatch, *overrides)
    config = SimpleNamespace(
        scenes=SimpleNamespace(
            clip_ids=clip_ids,
            suite_ids=None,
            suite_configs=[object()],
            scene_suite_dirs=[],
            local_usdz_dir=local_usdz_dir,
        )
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        update_scene_config(config)


def test_scene_suite_config_list_composes_from_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = Path(__file__).parents[1] / "configs"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        base_config = compose(config_name="base_config.yaml")
    config = OmegaConf.merge(
        base_config,
        {
            "scenes": {
                "clip_ids": None,
                "suite_configs": [
                    {
                        "csv_path": "data/scenes/scene_suites/public_2601.csv",
                        "t0_offset_us": -1_000_000,
                    }
                ],
            }
        },
    )
    _hydra_overrides(monkeypatch)
    update_scene_config(config)

    assert config.scenes.clip_ids is None
    assert len(config.scenes.suite_configs) == 1
    assert Path(config.scenes.suite_configs[0].csv_path).name == "public_2601.csv"
    assert config.scenes.suite_configs[0].t0_offset_us == -1_000_000


def test_composed_scene_suite_requires_clearing_default_clip_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hydra_overrides(monkeypatch)
    config = SimpleNamespace(
        scenes=SimpleNamespace(
            clip_ids=list(_DEFAULT_CLIP_IDS),
            suite_ids=None,
            suite_configs=[object()],
            scene_suite_dirs=[],
            local_usdz_dir=None,
        )
    )
    with pytest.raises(
        ValueError,
        match=re.escape("either scenes.suite_configs or scenes.clip_ids"),
    ):
        update_scene_config(config)


def test_base_clip_ids_match_inherited_default() -> None:
    config_dir = Path(__file__).parents[1] / "configs"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(config_name="base_config.yaml")
    assert tuple(config.scenes.clip_ids) == _DEFAULT_CLIP_IDS


def test_scene_selection_requires_one_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _hydra_overrides(monkeypatch)
    config = SimpleNamespace(
        scenes=SimpleNamespace(
            clip_ids=None,
            suite_ids=None,
            suite_configs=None,
            scene_suite_dirs=[],
            local_usdz_dir=None,
        )
    )

    with pytest.raises(ValueError, match="Set one scene selector"):
        update_scene_config(config)


def test_sideloaded_scene_suite_list_composes() -> None:
    config_dir = Path(__file__).parents[1] / "configs"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(
            config_name="base_config.yaml",
            overrides=[
                "scenes.suite_configs=[{csv_path:/tmp/first.csv,t0_offset_us:-10},"
                "{csv_path:/tmp/second.csv}]"
            ],
        )
    assert [suite.csv_path for suite in config.scenes.suite_configs] == [
        "/tmp/first.csv",
        "/tmp/second.csv",
    ]
    assert [suite.t0_offset_us for suite in config.scenes.suite_configs] == [-10, 0]


def test_local_directory_keeps_explicit_clip_filter(monkeypatch: pytest.MonkeyPatch):
    _hydra_overrides(
        monkeypatch,
        "scenes.local_usdz_dir = /tmp/clips",
        "scenes.clip_ids = [clip-a]",
    )
    config = SimpleNamespace(
        scenes=SimpleNamespace(
            clip_ids=["clip-a"],
            suite_ids=None,
            suite_configs=None,
            scene_suite_dirs=[],
            local_usdz_dir="/tmp/clips",
        )
    )
    update_scene_config(config)
    assert config.scenes.clip_ids == ["clip-a"]


@pytest.mark.parametrize(
    ("overrides", "clip_ids", "expected"),
    [
        (
            ("scenes.local_usdz_dir = /tmp/clips",),
            list(_DEFAULT_CLIP_IDS),
            None,
        ),
        (
            ("scenes.local_usdz_dir = /tmp/clips",),
            ["clip-from-composed-config"],
            ["clip-from-composed-config"],
        ),
        ((), ["clip-from-composed-config"], ["clip-from-composed-config"]),
    ],
)
def test_local_directory_selector_matrix(
    monkeypatch: pytest.MonkeyPatch,
    overrides: tuple[str, ...],
    clip_ids: list[str],
    expected: list[str] | None,
) -> None:
    _hydra_overrides(monkeypatch, *overrides)
    config = SimpleNamespace(
        scenes=SimpleNamespace(
            clip_ids=clip_ids,
            suite_ids=None,
            suite_configs=None,
            scene_suite_dirs=[],
            local_usdz_dir="/tmp/clips",
        )
    )
    update_scene_config(config)
    assert config.scenes.clip_ids == expected
