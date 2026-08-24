# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Structure and inference tests for the vendored DiffusionDrive model."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch
from navsim_diffusiondrive_challenge.simscale_diffusiondrive.config import (
    DiffusionDriveConfig,
)
from navsim_diffusiondrive_challenge.simscale_diffusiondrive.model import (
    DiffusionDriveModel,
)
from torch import nn

MODEL_PACKAGE = (
    Path(__file__).resolve().parents[1]
    / "navsim_diffusiondrive_challenge"
    / "simscale_diffusiondrive"
)
FORBIDDEN_MODEL_IMPORTS = ("navsim", "nuplan", "diffusers", "einops")


class FakeBackbone(nn.Module):
    """Returns the feature shapes consumed by the released decoder."""

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        batch_size = image.shape[0]
        kwargs = {"device": image.device, "dtype": image.dtype}
        return (
            torch.zeros((batch_size, 64, 64, 64), **kwargs),
            torch.zeros((batch_size, 512, 8, 8), **kwargs),
            None,
        )


def _anchor() -> np.ndarray:
    return np.linspace(-1.0, 1.0, 20 * 8 * 2, dtype=np.float32).reshape(20, 8, 2)


def _model() -> DiffusionDriveModel:
    return DiffusionDriveModel(
        DiffusionDriveConfig(), _anchor(), backbone=FakeBackbone()
    )


def test_released_model_structure_is_preserved() -> None:
    model = _model()

    assert tuple(model._trajectory_head.plan_anchor.shape) == (20, 8, 2)
    assert model._query_embedding.num_embeddings == 31
    assert len(model._trajectory_head.diff_decoder.layers) == 2


def test_forward_with_supplied_noise_returns_finite_trajectory() -> None:
    model = _model().eval()
    batch_size = 2
    features = {
        "camera_feature": torch.zeros((batch_size, 3, 32, 32), dtype=torch.float32),
        "status_feature": torch.zeros((batch_size, 8), dtype=torch.float32),
    }
    noise = torch.zeros((batch_size, 20, 8, 2), dtype=torch.float32)

    with torch.inference_mode():
        trajectory = model(features, noise=noise)

    assert trajectory.shape == (batch_size, 8, 3)
    assert torch.isfinite(trajectory).all()


@pytest.mark.parametrize(
    ("anchor", "message"),
    [
        (np.zeros((20, 8, 3), dtype=np.float32), "shape"),
        (np.zeros((20, 8, 2), dtype=np.float64), "float32"),
        (
            np.full((20, 8, 2), np.nan, dtype=np.float32),
            "finite",
        ),
    ],
)
def test_invalid_anchor_is_rejected(anchor: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DiffusionDriveModel(DiffusionDriveConfig(), anchor, backbone=FakeBackbone())


def test_model_runtime_imports_are_dependency_free() -> None:
    violations: list[str] = []
    for name in ("backbone.py", "blocks.py", "model.py"):
        path = MODEL_PACKAGE / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = [node.module]
            for module in modules:
                if module.startswith(FORBIDDEN_MODEL_IMPORTS):
                    violations.append(f"{name}: {module}")

    assert not violations, "Forbidden model imports:\n" + "\n".join(violations)


def test_upstream_and_vendored_models_match_released_fp32_inference() -> None:
    if os.environ.get("DIFFUSIONDRIVE_UPSTREAM_PARITY") != "1":
        pytest.skip("set DIFFUSIONDRIVE_UPSTREAM_PARITY=1 to run upstream parity")
    checkpoint = os.environ.get("DIFFUSIONDRIVE_REAL_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set DIFFUSIONDRIVE_REAL_CHECKPOINT to run upstream parity")

    repo_root = Path(__file__).resolve().parents[3]
    runner = textwrap.dedent(
        r"""
        import json
        import sys
        from pathlib import Path

        import numpy as np
        import torch
        import timm

        repo = Path(sys.argv[1])
        checkpoint_path = Path(sys.argv[2])
        reference = repo / "reference/SimScale"
        submission = repo / "e2e_challenge/sample_submission_simscale_navsim_diffusiondrive"
        sys.path.insert(0, str(reference))
        sys.path.insert(0, str(submission))

        original_create_model = timm.create_model

        def offline_create_model(*args, **kwargs):
            if kwargs.get("pretrained"):
                kwargs["pretrained"] = False
                kwargs.pop("pretrained_cfg_overlay", None)
            return original_create_model(*args, **kwargs)

        timm.create_model = offline_create_model

        from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
        from navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel
        from navsim_diffusiondrive_challenge.simscale_diffusiondrive.config import DiffusionDriveConfig
        from navsim_diffusiondrive_challenge.simscale_diffusiondrive.model import DiffusionDriveModel

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for upstream parity")
        device = torch.device("cuda")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        prefix = "agent._transfuser_model."
        state_dict = {
            key.removeprefix(prefix): value
            for key, value in payload["state_dict"].items()
        }
        anchor = state_dict["_trajectory_head.plan_anchor"].numpy().copy()
        noise = torch.linspace(-1.0, 1.0, 20 * 8 * 2, dtype=torch.float32).reshape(1, 20, 8, 2)
        features = {
            "camera_feature": torch.zeros((1, 3, 512, 2048), device=device),
            "status_feature": torch.zeros((1, 8), device=device),
        }

        upstream_config = TransfuserConfig()
        upstream_config.plan_anchor_path = str(
            reference / "traj_final/kmeans_navsim_traj_20.npy"
        )
        upstream = V2TransfuserModel(upstream_config)
        upstream.load_state_dict(state_dict, strict=True)
        upstream.to(device).eval()

        original_randn = torch.randn

        def fixed_randn(*args, **kwargs):
            shape = tuple(args[0]) if args else tuple(kwargs.get("size", ()))
            if shape == tuple(noise.shape):
                return noise.to(
                    device=kwargs.get("device", device),
                    dtype=kwargs.get("dtype", noise.dtype),
                )
            return original_randn(*args, **kwargs)

        torch.randn = fixed_randn
        with torch.inference_mode():
            upstream_output = upstream(features)["trajectory"].cpu()
        torch.randn = original_randn
        del upstream
        torch.cuda.empty_cache()

        vendored = DiffusionDriveModel(DiffusionDriveConfig(), anchor)
        vendored.load_state_dict(state_dict, strict=True)
        vendored.to(device).eval()
        with torch.inference_mode():
            vendored_output = vendored(features, noise=noise.to(device)).cpu()

        torch.testing.assert_close(
            vendored_output,
            upstream_output,
            rtol=1e-5,
            atol=1e-5,
        )
        print(json.dumps({
            "shape": list(vendored_output.shape),
            "max_abs_error": float((vendored_output - upstream_output).abs().max()),
            "finite": bool(torch.isfinite(vendored_output).all()),
        }, sort_keys=True))
        """
    )
    result = subprocess.run(
        [
            "/opt/conda/envs/navsim/bin/python",
            "-c",
            runner,
            str(repo_root),
            checkpoint,
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout.strip().splitlines()[-1])
    assert evidence["shape"] == [1, 8, 3]
    assert evidence["finite"] is True
    assert evidence["max_abs_error"] <= 1e-5
