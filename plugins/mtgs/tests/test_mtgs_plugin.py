# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for MTGS plugin discovery and configuration."""

from types import SimpleNamespace

import pytest


def test_mtgs_plugin_importable():
    """Verify the plugin package is importable."""
    import alpasim_mtgs
    from alpasim_mtgs.config import MTGSConfig
    from alpasim_mtgs.service import MTGSService

    assert alpasim_mtgs.__name__ == "alpasim_mtgs"
    assert MTGSService is not None
    assert MTGSConfig is not None


def test_mtgs_config_schema():
    """Verify config schema is exposed correctly."""
    from alpasim_mtgs.config import MTGSConfig
    from alpasim_mtgs.service import MTGSService

    schema = MTGSService.get_config_schema()
    assert schema is MTGSConfig


def test_mtgs_config_defaults():
    """Verify config has expected defaults."""
    from alpasim_mtgs.config import MTGSConfig

    config = MTGSConfig()
    assert config.skip_warmup is False


def test_mtgs_from_config_empty():
    """Verify from_config works with empty config dict."""
    from alpasim_mtgs.service import MTGSService

    service = MTGSService.from_config(
        raw_config={},
        address="localhost:8080",
        skip=True,
        camera_catalog=None,
    )
    assert service is not None
    assert service.skip is True
    assert service._mtgs_config.skip_warmup is False


def test_mtgs_from_config_with_options():
    """Verify from_config passes options correctly."""
    from alpasim_mtgs.service import MTGSService

    service = MTGSService.from_config(
        raw_config={"skip_warmup": True},
        address="localhost:9090",
        skip=False,
        camera_catalog=None,
    )
    assert service._mtgs_config.skip_warmup is True
    assert service.address == "localhost:9090"


def test_mtgs_plugin_registry():
    """Verify plugin is discoverable via PluginRegistry."""
    try:
        from alpasim_plugins import PluginRegistry

        registry = PluginRegistry("alpasim.services")
        service_class = registry.get("mtgs")
        assert service_class is not None

        from alpasim_mtgs.service import MTGSService

        assert service_class is MTGSService
    except ImportError:
        pytest.skip("alpasim_plugins not installed")


def test_mtgs_engine_importable():
    """Verify server-side engine code is importable."""
    try:
        from alpasim_mtgs.server.engine.base_renderer import BaseRenderer, RenderState
        from alpasim_mtgs.server.engine.utils.gaussian_utils import quat_to_rotmat
        from alpasim_mtgs.server.engine.utils.geometry_utils import Sim2

        assert BaseRenderer is not None
        assert RenderState is not None
        assert quat_to_rotmat is not None
        assert Sim2 is not None
    except ImportError as e:
        pytest.skip(f"Server dependencies not installed: {e}")


def _mtgs_user_config(
    *, extra_params: dict | None = None, smooth_trajectories: bool = False
):
    from alpasim_runtime.config import (
        SceneProviderConfig,
        TrajdataDatasetConfig,
        TrajdataProviderConfig,
        UserSimulatorConfig,
    )

    return UserSimulatorConfig(
        scene_provider=SceneProviderConfig(
            kind="trajdata",
            usdz=None,
            trajdata=TrajdataProviderConfig(
                cache_location="/tmp/trajdata-cache",
                desired_dt=0.1,
                load_vector_map=True,
                dataset=TrajdataDatasetConfig(
                    name="nuplan_test",
                    data_dir="/tmp/nuplan",
                    extra_params=extra_params or {},
                ),
            ),
        ),
        smooth_trajectories=smooth_trajectories,
    )


def test_mtgs_scene_loader_requires_asset_base_path(monkeypatch):
    from alpasim_mtgs.server import main as mtgs_main

    monkeypatch.setattr(mtgs_main, "TRAJDATA_AVAILABLE", True)

    with pytest.raises(ValueError, match="asset_base_path"):
        mtgs_main.create_get_scene_function(_mtgs_user_config())


def test_mtgs_scene_loader_uses_public_trajdata_dataset_api(monkeypatch):
    from alpasim_mtgs.server import main as mtgs_main

    captured = {}
    scene_cache = object()
    map_api = object()

    class FakeScene:
        name = "scene-a"
        env_name = "nuplan_test"

    class FakeUnifiedDataset:
        vector_map_params = {"incl_road_lanes": True}

        def __init__(self, **params):
            captured["dataset_params"] = params
            self._scene = FakeScene()

        @property
        def scene_name_to_index(self):
            return {"scene-a": 0}

        @property
        def map_api(self):
            return map_api

        def get_scene_cache(self, scene):
            captured["cache_scene"] = scene
            return scene_cache

        def num_scenes(self):
            return 1

        def get_scene(self, idx):
            assert idx == 0
            return self._scene

    def fake_trajdata_data_source(**kwargs):
        captured["data_source_kwargs"] = kwargs
        return SimpleNamespace(asset_path="/tmp/mtgs-assets/navtest/assets/scene-a")

    monkeypatch.setattr(mtgs_main, "TRAJDATA_AVAILABLE", True)
    monkeypatch.setattr(mtgs_main, "UnifiedDataset", FakeUnifiedDataset)
    monkeypatch.setattr(mtgs_main, "TrajdataDataSource", fake_trajdata_data_source)

    get_scene, get_available_scene_ids = mtgs_main.create_get_scene_function(
        _mtgs_user_config(extra_params={"asset_base_path": "/tmp/mtgs-assets"})
    )

    assert get_available_scene_ids() == ["scene-a"]
    data_source = get_scene("scene-a")

    assert data_source.asset_path == "/tmp/mtgs-assets/navtest/assets/scene-a"
    assert captured["dataset_params"]["desired_data"] == ["nuplan_test"]
    assert captured["dataset_params"]["dataset_kwargs"] == {
        "nuplan_test": {"asset_base_path": "/tmp/mtgs-assets"}
    }
    assert captured["cache_scene"].name == "scene-a"

    kwargs = captured["data_source_kwargs"]
    assert kwargs["scene"].name == "scene-a"
    assert kwargs["scene_cache"] is scene_cache
    assert kwargs["map_api"] is map_api
    assert kwargs["vector_map_params"] == {"incl_road_lanes": True}
    assert kwargs["asset_base_path"] == "/tmp/mtgs-assets/navtest/assets"
