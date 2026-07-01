# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Integration tests for the trafficsim CATK model.

Run with:
    uv run pytest src/trafficsim/tests/test_catk_integration.py -v -s

Tests:
    test_catk_synthetic  — Loads CATK model + weights, runs inference on synthetic
                           env_data (no USDZ needed). Requires CUDA + torch-cluster +
                           model weights in data/trafficsim-models/.

    test_catk_scene_adapter — Loads a ClipGT USDZ through the trafficsim decoder.
                           Requires a ClipGT-format USDZ (set TRAFFICSIM_TEST_USDZ).

    test_catk_inference  — Full end-to-end: ClipGT USDZ → CATK model → predictions.
                           Requires both USDZ data and model weights.

Environment variables:
    TRAFFICSIM_MODELS_DIR  — Path to model weights (default: data/trafficsim-models)
    TRAFFICSIM_TEST_USDZ   — Path to a single ClipGT .usdz file
    TRAFFICSIM_TEST_USDZ_DIR — Directory containing ClipGT .usdz files
"""


import os
from pathlib import Path

import pytest
import torch
from alpasim_trafficsim.catk.obstacle_classes import obstacle_class_metadata

MAP_ELEMENT_NAME2_TYPEID = {
    "lane_lines": 0,
    "road_boundaries": 1,
}

REPO_ROOT = Path(__file__).resolve().parents[3]

MODELS_DIR = Path(
    os.environ.get("TRAFFICSIM_MODELS_DIR", REPO_ROOT / "data" / "trafficsim-models")
)

_usdz_env = os.environ.get("TRAFFICSIM_TEST_USDZ")
_usdz_dir_env = os.environ.get("TRAFFICSIM_TEST_USDZ_DIR")


def _find_usdz() -> Path | None:
    if _usdz_env:
        p = Path(_usdz_env)
        return p if p.is_file() else None
    if _usdz_dir_env:
        d = Path(_usdz_dir_env)
    else:
        d = REPO_ROOT / "data" / "trafficsim-test-data" / "usdz"
    if d.is_dir():
        usdz_files = sorted(d.glob("*.usdz"))
        return usdz_files[0] if usdz_files else None
    return None


_test_usdz = _find_usdz()

_skip_no_usdz = pytest.mark.skipif(
    _test_usdz is None,
    reason="No ClipGT USDZ files found. Set TRAFFICSIM_TEST_USDZ or TRAFFICSIM_TEST_USDZ_DIR.",
)
_skip_no_weights = pytest.mark.skipif(
    not (MODELS_DIR / "catk_v120" / "latest.ckpt").is_file(),
    reason=f"CATK weights not found in {MODELS_DIR}",
)
_skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


def _try_import_torch_cluster() -> bool:
    try:
        __import__("torch_cluster")

        return True
    except ImportError:
        return False


_skip_no_torch_cluster = pytest.mark.skipif(
    not _try_import_torch_cluster(),
    reason="torch-cluster not installed (requires matching CUDA toolkit)",
)


def _make_synthetic_env_data(
    num_agents: int = 3,
    num_timesteps: int = 80,
    num_history: int = 16,
    num_polylines: int = 20,
    num_points_per_polyline: int = 51,
) -> dict:
    """Build a minimal env_data dict with plausible shapes for CATK.

    Creates a straight-road scene with the ego driving forward and a few
    agents at fixed offsets.
    """
    dt = 0.1  # 10 Hz
    times = torch.arange(num_timesteps, dtype=torch.float32) * dt

    # Ego drives forward along +x at ~10 m/s
    ego_xyz = torch.zeros(num_timesteps, 3)
    ego_xyz[:, 0] = times * 10.0  # x = velocity * t
    ego_heading = torch.zeros(num_timesteps)  # heading = 0 (along +x)
    ego_lwh = torch.tensor([4.5, 2.0, 1.6])

    # Agents: offset laterally and moving forward at slightly different speeds
    agents_xyz = torch.zeros(num_agents, num_timesteps, 3)
    agents_heading = torch.zeros(num_agents, num_timesteps)
    agents_valid = torch.ones(num_agents, num_timesteps, dtype=torch.bool)
    agents_lwh = (
        torch.tensor([4.5, 2.0, 1.6]).unsqueeze(0).expand(num_agents, 3).clone()
    )

    for i in range(num_agents):
        lateral_offset = (i + 1) * 3.5  # 3.5m lane width
        speed = 8.0 + i * 2.0
        agents_xyz[i, :, 0] = times * speed + 10.0  # ahead of ego
        agents_xyz[i, :, 1] = lateral_offset

    # Map: parallel lane lines along +x
    map_data = {}
    polylines_per_element = num_polylines // 2

    for element_name, map_type_name in [
        ("lanelines", "lane_lines"),
        ("road_boundaries", "road_boundaries"),
    ]:
        label_id = MAP_ELEMENT_NAME2_TYPEID[map_type_name]
        polylines = torch.zeros(polylines_per_element, num_points_per_polyline, 3)
        for j in range(polylines_per_element):
            xs = torch.linspace(j * 20.0, (j + 1) * 20.0, num_points_per_polyline)
            polylines[j, :, 0] = xs
            polylines[j, :, 1] = (j % 4) * 3.5  # lane offsets

        map_data[element_name] = {
            "polylines": polylines,
            "label": [label_id] * polylines_per_element,
        }

    return {
        "ego": {
            "xyz": ego_xyz,
            "heading": ego_heading,
            "lwh": ego_lwh,
        },
        "agents": {
            "xyz": agents_xyz,
            "heading": agents_heading,
            "valid_mask": agents_valid,
            "lwh": agents_lwh,
            "track_ids": torch.arange(num_agents),
            "class_ids": torch.zeros(num_agents, dtype=torch.long),  # all "car"
            "num_obstacles": num_agents,
        },
        "map": map_data,
        "env": {
            "curr_t": num_history - 1,
        },
        "metadata": {
            "frame_rate": 10,
            **obstacle_class_metadata(),
        },
    }


def test_freeze_agent_data_includes_static_agents_and_ego() -> None:
    from alpasim_trafficsim.catk.env_data_adapter import build_freeze_agent_data

    env_data = _make_synthetic_env_data(num_agents=3, num_timesteps=32)
    curr_t = int(env_data["env"]["curr_t"])
    env_data["env"]["agent_is_static"] = [False, True, True]
    env_data["agents"]["valid_mask"][2, curr_t] = False
    env_data["agents"]["xyz"][1, curr_t] = torch.tensor([11.0, 12.0, 13.0])
    env_data["agents"]["heading"][1, curr_t] = torch.tensor(0.5)

    freeze_data, freeze_mask = build_freeze_agent_data(
        env_data,
        curr_t=curr_t,
        target_steps=6,
        dt=0.1,
        device="cpu",
    )

    assert freeze_mask.tolist() == [False, True, False, True]
    assert freeze_data["num_obstacles"].item() == 2
    torch.testing.assert_close(
        freeze_data["agent"]["position"][0],
        torch.tensor([[11.0, 12.0, 13.0]]).expand(6, 3),
    )
    torch.testing.assert_close(
        freeze_data["agent"]["heading"][0],
        torch.full((6,), 0.5),
    )
    assert freeze_data["agent"]["id"].tolist() == [1, 0]


@_skip_no_torch_cluster
def test_world_model_passes_static_and_ego_freeze_mask_to_decoder() -> None:
    from alpasim_trafficsim.catk.model_adapter import CATK

    class FakeAgentEncoder:
        num_historical_steps = 16

        def __init__(self) -> None:
            self.num_future_steps = 0
            self.captured: dict[str, object] = {}

        def inference_with_mask(
            self,
            *,
            tokenized_agent,
            map_feature: object,
            sampling_scheme: object,
            freeze_agent_future,
            freeze_agent_mask,
            freeze_tokenized_agent,
        ):
            _ = map_feature, sampling_scheme
            self.captured = {
                "freeze_agent_future": freeze_agent_future,
                "freeze_agent_mask": freeze_agent_mask.clone(),
                "freeze_tokenized_agent": freeze_tokenized_agent,
            }
            n_agent = int(tokenized_agent["ego_mask"].shape[0])
            n_step = int(self.num_future_steps)
            return {
                "pred_traj_10hz": torch.zeros((n_agent, n_step, 2)),
                "pred_z_10hz": torch.zeros((n_agent, n_step)),
                "pred_head_10hz": torch.zeros((n_agent, n_step)),
                "pred_valid_10hz": torch.ones((n_agent, n_step), dtype=torch.bool),
            }

    class FakeEncoder:
        def __init__(self) -> None:
            self.agent_encoder = FakeAgentEncoder()

        @staticmethod
        def map_encoder(_tokenized_map):
            return {}

    class FakeTokenProcessor:
        def __init__(self) -> None:
            self.freeze_agent_ids: list[int] = []

        def __call__(
            self,
            state,
            *,
            apply_heading_correction: bool,
            apply_boundary_extrapolation: bool,
        ):
            _ = apply_heading_correction, apply_boundary_extrapolation
            n_agent = int(state["agent"]["id"].shape[0])
            ego_mask = torch.zeros((n_agent,), dtype=torch.bool)
            ego_mask[-1] = True
            return {}, {"ego_mask": ego_mask}, None, None

        def tokenize_agent(
            self,
            data,
            *,
            apply_heading_correction: bool,
            apply_boundary_extrapolation: bool,
        ):
            _ = apply_heading_correction, apply_boundary_extrapolation
            self.freeze_agent_ids = data["agent"]["id"].detach().cpu().tolist()
            return {
                "gt_pos_raw": data["agent"]["position"][..., :2],
                "gt_head_raw": data["agent"]["heading"],
                "gt_valid_raw": data["agent"]["valid_mask"],
                "gt_idx": torch.zeros_like(data["agent"]["heading"], dtype=torch.long),
            }, {}

    class FakeModel:
        def __init__(self) -> None:
            self.encoder = FakeEncoder()
            self.token_processor = FakeTokenProcessor()
            self.validation_rollout_sampling = {}

    env_data = _make_synthetic_env_data(num_agents=3, num_timesteps=32)
    env_data["env"]["agent_is_static"] = [False, True, False]

    catk = CATK.__new__(CATK)
    catk.device = "cpu"
    catk.model_input_step_num = 16
    catk.model_predict_step_num = 5
    catk.delta_t = 0.1
    catk.use_downsampled_lines = False
    catk.disable_sub_plyline_type = True
    catk.model = FakeModel()

    input_data = catk.create_model_input(
        env_data,
        filter_map_by_ego=False,
        filter_distance_th=0.0,
    )["input_data"]
    catk.inference(input_data)

    agent_encoder = catk.model.encoder.agent_encoder
    assert agent_encoder.captured["freeze_agent_future"] is True
    assert agent_encoder.captured["freeze_agent_mask"].tolist() == [
        False,
        True,
        False,
        True,
    ]
    assert catk.model.token_processor.freeze_agent_ids == [1, 0]


@pytest.mark.integration
@_skip_no_weights
@_skip_no_cuda
@_skip_no_torch_cluster
def test_catk_synthetic():
    """Load CATK model and run inference on synthetic data (no USDZ needed)."""
    from alpasim_trafficsim.catk.model_adapter import CATK

    env_data = _make_synthetic_env_data()

    catk_dir = MODELS_DIR / "catk_v120"
    model = CATK(
        config_path=str(catk_dir / "config.yaml"),
        ckpt_path=str(catk_dir / "latest.ckpt"),
        token_pkl_dir=str(MODELS_DIR / "tokens"),
        disable_sub_plyline_type=True,
        device="cuda",
    )

    result = model.create_model_input(
        env_data, filter_map_by_ego=True, filter_distance_th=100.0
    )
    input_data = result["input_data"]

    actions = model.inference(input_data)

    assert "agent_future_xyz" in actions
    assert "agent_future_heading" in actions
    assert "agent_future_valid_mask" in actions

    num_agents = actions["agent_future_xyz"].shape[0]
    num_steps = actions["agent_future_xyz"].shape[1]

    assert actions["agent_future_heading"].shape == (num_agents, num_steps)
    assert actions["agent_future_valid_mask"].shape == (num_agents, num_steps)

    valid = actions["agent_future_valid_mask"].bool()
    if valid.any():
        valid_xyz = actions["agent_future_xyz"][valid]
        assert torch.isfinite(
            valid_xyz
        ).all(), "Non-finite positions in valid predictions"


# ---------------------------------------------------------------------------
# Tests that require ClipGT USDZ files
# ---------------------------------------------------------------------------


@pytest.mark.integration
@_skip_no_usdz
def test_catk_scene_adapter():
    """Sanity-check: load a ClipGT USDZ and verify the env_data structure."""
    from alpasim_runtime.scene_loader import ArtifactSceneProvider, SceneLoader
    from alpasim_trafficsim.catk.scene_adapter import CATKSceneAdapter

    usdz_path = _test_usdz

    scene_loader = SceneLoader(
        ArtifactSceneProvider.from_path(
            usdz_path,
            smooth_trajectories=False,
        )
    )
    scene_id = next(iter(scene_loader.scene_ids))
    env_data = CATKSceneAdapter().load(scene_loader.get_data_source(scene_id))

    assert "map" in env_data
    assert "ego" in env_data
    assert "agents" in env_data
    assert "env" in env_data

    assert env_data["ego"]["xyz"].ndim == 2
    assert env_data["ego"]["xyz"].shape[1] == 3
    assert env_data["ego"]["heading"].ndim == 1
    assert env_data["env"]["curr_t"] >= 0


@pytest.mark.integration
@_skip_no_usdz
@_skip_no_weights
@_skip_no_cuda
@_skip_no_torch_cluster
def test_catk_inference():
    """Full end-to-end: ClipGT USDZ → CATK model → predictions."""
    from alpasim_runtime.scene_loader import ArtifactSceneProvider, SceneLoader
    from alpasim_trafficsim.catk.model_adapter import CATK
    from alpasim_trafficsim.catk.scene_adapter import CATKSceneAdapter

    usdz_path = _test_usdz

    scene_loader = SceneLoader(
        ArtifactSceneProvider.from_path(
            usdz_path,
            smooth_trajectories=False,
        )
    )
    scene_id = next(iter(scene_loader.scene_ids))
    env_data = CATKSceneAdapter(
        num_history_steps=16,
        motion_stepsize=0.1,
    ).load(scene_loader.get_data_source(scene_id))

    catk_dir = MODELS_DIR / "catk_v120"
    model = CATK(
        config_path=str(catk_dir / "config.yaml"),
        ckpt_path=str(catk_dir / "latest.ckpt"),
        token_pkl_dir=str(MODELS_DIR / "tokens"),
        disable_sub_plyline_type=True,
        device="cuda",
    )

    result = model.create_model_input(
        env_data, filter_map_by_ego=True, filter_distance_th=100.0
    )
    input_data = result["input_data"]

    actions = model.inference(input_data)

    assert "agent_future_xyz" in actions
    assert "agent_future_heading" in actions
    assert "agent_future_valid_mask" in actions

    num_agents = actions["agent_future_xyz"].shape[0]
    num_steps = actions["agent_future_xyz"].shape[1]

    assert actions["agent_future_heading"].shape == (num_agents, num_steps)
    assert actions["agent_future_valid_mask"].shape == (num_agents, num_steps)

    valid = actions["agent_future_valid_mask"].bool()
    if valid.any():
        valid_xyz = actions["agent_future_xyz"][valid]
        assert torch.isfinite(
            valid_xyz
        ).all(), "Non-finite positions in valid predictions"
