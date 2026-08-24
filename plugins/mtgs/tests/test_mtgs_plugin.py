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
    assert config.skip_warmup is True


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
    assert service._mtgs_config.skip_warmup is True


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


def _rigid_model_for_log_pose_tests(*, log_replay: bool = True):
    import torch
    from alpasim_mtgs.server.engine.gaussian_model.rigid_object import (
        RigidPortableSubModel,
    )

    model = RigidPortableSubModel.__new__(RigidPortableSubModel)
    torch.nn.Module.__init__(model)
    model.log_replay = log_replay
    model.static_in_log = False
    model.gauss_params = {"means": torch.zeros(1)}
    model.log_start_time = 1_621_970_015_499_931
    model.log_timestamps = torch.tensor([0.0, 500_000.0, 1_000_000.0])
    model.log_quats = torch.nn.Parameter(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    model.log_trans = torch.nn.Parameter(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ]
        )
    )
    return model


def test_rigid_log_pose_accepts_relative_and_absolute_timestamps():
    import torch

    model = _rigid_model_for_log_pose_tests()
    relative_timestamp = 250_000
    absolute_timestamp = model.log_start_time + relative_timestamp

    _, relative_trans, _ = model._get_log_pose_from_timestamp(relative_timestamp)
    _, absolute_trans, _ = model._get_log_pose_from_timestamp(absolute_timestamp)

    torch.testing.assert_close(relative_trans, torch.tensor([2.5, 0.0, 0.0]))
    torch.testing.assert_close(absolute_trans, relative_trans)


@pytest.mark.parametrize(
    ("timestamp", "expected_x"),
    [
        (-100_000, 0.0),
        (1_500_000, 10.0),
        (1_621_970_015_399_931, 0.0),
        (1_621_970_016_999_931, 10.0),
    ],
)
def test_rigid_log_pose_clamps_out_of_range_timestamps(timestamp, expected_x):
    model = _rigid_model_for_log_pose_tests()

    _, trans, _ = model._get_log_pose_from_timestamp(timestamp)

    assert trans[0].item() == expected_x


def test_rigid_dynamic_actor_without_pose_requires_log_replay():
    model = _rigid_model_for_log_pose_tests(log_replay=False)

    assert model._decide_global_pose(timestamp=0) is None
    assert model.get_global_gaussians(timestamp=0) is None

    model.log_replay = True
    assert model._decide_global_pose(timestamp=0) is not None


def test_rigid_static_actor_uses_log_pose_without_log_replay():
    import torch

    model = _rigid_model_for_log_pose_tests(log_replay=False)
    model.static_in_log = True
    model.log_quats = torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    model.log_trans = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))

    quat, trans, timestamp = model._decide_global_pose(timestamp=0)

    torch.testing.assert_close(quat, model.log_quats)
    torch.testing.assert_close(trans, model.log_trans)
    assert timestamp == 0


def test_rigid_explicit_pose_overrides_static_log_pose():
    import torch

    model = _rigid_model_for_log_pose_tests(log_replay=False)
    model.static_in_log = True
    model.log_quats = torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    model.log_trans = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
    requested_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
    requested_trans = torch.tensor([5.0, 6.0, 7.0])

    quat, trans, _ = model._decide_global_pose(
        quat=requested_quat,
        trans=requested_trans,
        timestamp=0,
    )

    torch.testing.assert_close(quat, requested_quat)
    torch.testing.assert_close(trans, requested_trans)


def test_mtgs_rgb_collection_matches_active_geometry():
    import torch
    from alpasim_mtgs.server.engine.mtgs import MTGS

    class ActiveModel:
        def get_global_gaussians(self, **_kwargs):
            return {
                "means": torch.zeros((1, 3)),
                "scales": torch.ones((1, 3)),
                "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                "opacities": torch.ones(1),
            }

        def get_gaussian_rgbs(self, **_kwargs):
            return torch.ones((1, 3))

    class InactiveModel:
        def get_global_gaussians(self, **_kwargs):
            return None

        def get_gaussian_rgbs(self, **_kwargs):
            raise AssertionError("inactive model must not contribute RGB features")

    renderer = MTGS.__new__(MTGS)
    renderer.device = torch.device("cpu")
    renderer.node_types = {"background": "background", "inactive": "na"}
    renderer.submodel_names = {
        "background": "background",
        "inactive": "inactive",
    }
    renderer.gaussian_models = {
        "background": ActiveModel(),
        "inactive": InactiveModel(),
    }
    renderer._world_cache_key = None
    renderer._active_asset_tokens = ()

    renderer.update_world(timestamp=0, agent_states={})
    renderer.update_gaussian_rgbs(torch.eye(4).unsqueeze(0))

    assert renderer._active_asset_tokens == ("background",)
    assert renderer.collected_gaussians["means"].shape[0] == 1
    assert renderer.collected_gaussians["rgbs"].shape[1] == 1


def test_mtgs_asset_manager_allows_missing_road_height_map(tmp_path, monkeypatch):
    import torch
    from alpasim_mtgs.server.engine.mtgs import MTGSAssetManager

    asset_id = "legacy-asset"
    background_dir = tmp_path / asset_id / "background"
    background_dir.mkdir(parents=True)
    (background_dir / f"{asset_id}.ckpt").touch()
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})

    manager = MTGSAssetManager(tmp_path, torch.device("cpu"))
    manager.reset(asset_id)

    assert manager.road_height_map is None


def _mtgs_user_config(
    *,
    extra_params: dict | None = None,
    smooth_trajectories: bool = False,
    vector_map_params: dict | None = None,
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
                vector_map_params=vector_map_params or {},
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


def test_build_token_to_asset_folder_infers_road_block_name(tmp_path):
    from alpasim_mtgs.server import main as mtgs_main

    (tmp_path / "scene.yaml").write_text(
        "central_log: log-a\n" "central_tokens:\n" "  - token-1\n" "  - token-2\n"
    )

    mapping = mtgs_main._build_token_to_asset_folder(tmp_path)

    assert mapping == {
        "token-1": "log-a-token-1",
        "token-2": "log-a-token-1",
    }


def test_mtgs_scene_loader_uses_public_trajdata_dataset_api(monkeypatch):
    from alpasim_mtgs.server import main as mtgs_main

    captured = {}
    scene_cache = object()
    map_api = object()

    class FakeScene:
        name = "scene-a"
        env_name = "nuplan_test"

    class FakeUnifiedDataset:
        def __init__(self, **params):
            captured["dataset_params"] = params
            self.vector_map_params = params.get("vector_map_params", {})
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
        resolver = kwargs.get("asset_folder_resolver")
        asset_folder = (
            resolver(kwargs["scene"]) if resolver is not None else kwargs["scene"].name
        )
        return SimpleNamespace(
            asset_path=f"/tmp/mtgs-assets/navtest/assets/{asset_folder}"
        )

    monkeypatch.setattr(mtgs_main, "TRAJDATA_AVAILABLE", True)
    monkeypatch.setattr(mtgs_main, "UnifiedDataset", FakeUnifiedDataset)
    monkeypatch.setattr(mtgs_main, "TrajdataDataSource", fake_trajdata_data_source)
    monkeypatch.setattr(
        mtgs_main,
        "_build_token_to_asset_folder",
        lambda _configs_dir: {"a": "asset-auto"},
    )

    get_scene, get_available_scene_ids = mtgs_main.create_get_scene_function(
        _mtgs_user_config(
            extra_params={
                "asset_base_path": "/tmp/mtgs-assets",
                "asset_folder_map": {"scene-a": "asset-b"},
            },
            vector_map_params={"incl_road_lanes": True, "incl_road_edges": True},
        )
    )

    assert get_available_scene_ids() == ["scene-a"]
    data_source = get_scene("scene-a")

    assert data_source.asset_path == "/tmp/mtgs-assets/navtest/assets/asset-b"
    assert captured["dataset_params"]["desired_data"] == ["nuplan_test"]
    assert captured["dataset_params"]["dataset_kwargs"] == {
        "nuplan_test": {
            "asset_base_path": "/tmp/mtgs-assets",
            "asset_folder_map": {"scene-a": "asset-b"},
        }
    }
    assert captured["cache_scene"].name == "scene-a"

    kwargs = captured["data_source_kwargs"]
    assert kwargs["scene"].name == "scene-a"
    assert kwargs["scene_cache"] is scene_cache
    assert kwargs["map_api"] is map_api
    assert kwargs["vector_map_params"] == {
        "incl_road_lanes": True,
        "incl_road_edges": True,
    }
    assert kwargs["asset_base_path"] == "/tmp/mtgs-assets/navtest/assets"
    assert kwargs["asset_folder_resolver"](kwargs["scene"]) == "asset-b"

    get_unmapped_scene, _ = mtgs_main.create_get_scene_function(
        _mtgs_user_config(extra_params={"asset_base_path": "/tmp/mtgs-assets"})
    )
    assert (
        get_unmapped_scene("scene-a").asset_path
        == "/tmp/mtgs-assets/navtest/assets/asset-auto"
    )


def test_mtgs_scene_loader_cache_is_bounded(monkeypatch):
    from alpasim_mtgs.server import main as mtgs_main

    created_scene_ids = []

    class FakeScene:
        env_name = "nuplan_test"

        def __init__(self, name):
            self.name = name

    class FakeUnifiedDataset:
        def __init__(self, **params):
            self.vector_map_params = params.get("vector_map_params", {})
            self._scenes = [FakeScene(f"scene-{idx}") for idx in range(3)]

        @property
        def scene_name_to_index(self):
            return {scene.name: idx for idx, scene in enumerate(self._scenes)}

        @property
        def map_api(self):
            return None

        def get_scene_cache(self, scene):
            return object()

        def num_scenes(self):
            return len(self._scenes)

        def get_scene(self, idx):
            return self._scenes[idx]

    def fake_trajdata_data_source(**kwargs):
        scene_id = kwargs["scene"].name
        created_scene_ids.append(scene_id)
        return SimpleNamespace(asset_path=f"/tmp/mtgs-assets/navtest/assets/{scene_id}")

    monkeypatch.setattr(mtgs_main, "TRAJDATA_AVAILABLE", True)
    monkeypatch.setattr(mtgs_main, "UnifiedDataset", FakeUnifiedDataset)
    monkeypatch.setattr(mtgs_main, "TrajdataDataSource", fake_trajdata_data_source)

    get_scene, _ = mtgs_main.create_get_scene_function(
        _mtgs_user_config(extra_params={"asset_base_path": "/tmp/mtgs-assets"}),
        cache_size=2,
    )

    first_scene = get_scene("scene-0")
    assert get_scene("scene-0") is first_scene
    get_scene("scene-1")
    get_scene("scene-2")

    assert get_scene.cache_info().currsize == 2
    assert get_scene("scene-0") is not first_scene
    assert created_scene_ids == ["scene-0", "scene-1", "scene-2", "scene-0"]
