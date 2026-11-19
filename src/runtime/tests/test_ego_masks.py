# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from alpasim_runtime import runtime_context
from alpasim_runtime.config import RendererConfig, SimulationConfig
from alpasim_runtime.ego_masks import load_ego_mask_overrides
from omegaconf import OmegaConf


def test_large_external_map_is_not_serialized_into_config(tmp_path):
    path = tmp_path / "overrides.yaml"
    entry = {"rig_config_id": "bank", "camera_logical_ids": ["front", "rear"]}
    path.write_text(yaml.safe_dump({f"clip-{i}": entry for i in range(18_000)}))
    cfg = OmegaConf.structured(SimulationConfig(ego_mask_overrides_file=str(path)))
    saved = OmegaConf.to_yaml(cfg, resolve=True)
    assert len(saved) < 10_000
    assert "clip-17999" not in saved
    assert OmegaConf.create(saved).ego_mask_overrides_file == str(path)
    with patch.object(Path, "open", autospec=True, side_effect=Path.open) as opened:
        first = load_ego_mask_overrides(str(path))
    assert len(first) == 18_000
    assert first["clip-17999"].rig_config_id == "bank"
    assert first["clip-17999"].camera_logical_ids == ("front", "rear")
    assert opened.call_count == 1


@pytest.mark.parametrize(
    "mapping",
    [
        None,
        [],
        {123: {}},
        {"clip": {}},
        {"clip": {"rig_config_id": "", "camera_logical_ids": ["front"]}},
        {"clip": {"rig_config_id": "bank", "camera_logical_ids": []}},
        {"clip": {"rig_config_id": "bank", "camera_logical_ids": ["front", "front"]}},
        {"clip": {"rig_config_id": "bank", "camera_logical_ids": "front"}},
        {"clip": {"rig_config_id": "bank", "camera_logical_ids": [{}]}},
    ],
)
def test_malformed_override_file_is_not_treated_as_missing_clip(tmp_path, mapping):
    path = tmp_path / "overrides.yaml"
    path.write_text(yaml.safe_dump(mapping))
    with pytest.raises(ValueError):
        load_ego_mask_overrides(str(path))


def test_missing_override_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ego_mask_overrides(str(tmp_path / "missing.yaml"))


@pytest.mark.parametrize(
    "clip_id",
    [
        "001906e6-c501-49a3-ad19-c252ded9f4f4",
        "clipgt-001906e6-c501-49a3-ad19-c252ded9f4f4@t0=AUTO",
        "../clip",
        "clip/name",
        " clip ",
    ],
)
def test_recognizable_invalid_source_ids_are_rejected(tmp_path, clip_id):
    path = tmp_path / "overrides.yaml"
    path.write_text(
        yaml.safe_dump(
            {clip_id: {"rig_config_id": "bank", "camera_logical_ids": ["front"]}}
        )
    )
    with pytest.raises(ValueError):
        load_ego_mask_overrides(str(path))


@pytest.mark.parametrize(
    "clip_id",
    ["dataset-item", "clipgt-a", "clipgt-001906e6-c501-49a3-ad19-c252ded9f4f4"],
)
def test_provider_source_ids_remain_supported(tmp_path, clip_id):
    path = tmp_path / "overrides.yaml"
    path.write_text(
        yaml.safe_dump(
            {clip_id: {"rig_config_id": "bank", "camera_logical_ids": ["front"]}}
        )
    )
    assert load_ego_mask_overrides(str(path))[clip_id].rig_config_id == "bank"


@pytest.mark.asyncio
@pytest.mark.parametrize("contents", [None, "not a mapping", "clip: {}"])
async def test_invalid_override_fails_startup_before_service_probing(
    tmp_path, monkeypatch, contents
):
    path = tmp_path / "overrides.yaml"
    if contents is not None:
        path.write_text(contents)
    config = SimpleNamespace(
        user=SimpleNamespace(
            simulation_config=SimulationConfig(ego_mask_overrides_file=str(path))
        )
    )
    monkeypatch.setattr(runtime_context, "parse_simulator_config", lambda *args: config)
    monkeypatch.setattr(runtime_context, "typed_parse_config", lambda *args: object())
    probe = AsyncMock()
    monkeypatch.setattr(runtime_context, "gather_versions_from_addresses", probe)
    with pytest.raises((ValueError, FileNotFoundError)):
        await runtime_context.build_runtime_context(
            user_config_path="user",
            network_config_path="network",
            eval_config_path="eval",
        )
    probe.assert_not_awaited()


@pytest.mark.asyncio
async def test_startup_passes_one_parsed_snapshot_and_logs_coverage(
    tmp_path, monkeypatch, caplog
):
    path = tmp_path / "overrides.yaml"
    path.write_text(
        "dataset-item:\n  rig_config_id: bank\n  camera_logical_ids: [front]\n"
    )
    config = SimpleNamespace(
        user=SimpleNamespace(
            simulation_config=SimulationConfig(ego_mask_overrides_file=str(path)),
            scenes=[],
            endpoints=SimpleNamespace(startup_timeout_s=1),
            renderer=RendererConfig(),
        ),
        network=object(),
    )
    monkeypatch.setattr(runtime_context, "parse_simulator_config", lambda *args: config)
    monkeypatch.setattr(runtime_context, "typed_parse_config", lambda *args: object())
    monkeypatch.setattr(runtime_context, "gather_versions_from_addresses", AsyncMock())
    monkeypatch.setattr(runtime_context, "validate_scenarios", AsyncMock())
    monkeypatch.setattr(
        runtime_context,
        "build_data_source_loader",
        lambda _: SimpleNamespace(data_source_infos=[]),
    )
    monkeypatch.setattr(runtime_context, "create_address_pools", lambda _: {})
    monkeypatch.setattr(runtime_context, "compute_max_in_flight", lambda *args: 1)
    with (
        caplog.at_level(logging.INFO),
        patch.object(Path, "open", autospec=True, side_effect=Path.open) as opened,
    ):
        context = await runtime_context.build_runtime_context(
            user_config_path="user",
            network_config_path="network",
            eval_config_path="eval",
        )
    assert opened.call_count == 1
    path.unlink()
    assert context.ego_mask_overrides["dataset-item"].rig_config_id == "bank"
    assert "Loaded 1 ego mask overrides" in caplog.text
    assert "0/0 configured source clips covered" in caplog.text


@pytest.mark.parametrize("field", ["rig_config_id", "camera_logical_ids"])
@pytest.mark.parametrize(
    "value", ["../bank", "a/b", "a\\b", ".", "..", " front", "front ", "", None, {}]
)
def test_invalid_bank_and_camera_ids_fail_loading(tmp_path, field, value):
    entry = {"rig_config_id": "bank", "camera_logical_ids": ["front"]}
    entry[field] = [value] if field == "camera_logical_ids" else value
    path = tmp_path / "overrides.yaml"
    path.write_text(yaml.safe_dump({"clip": entry}))
    with pytest.raises(ValueError):
        load_ego_mask_overrides(str(path))
