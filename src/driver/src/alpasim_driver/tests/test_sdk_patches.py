# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Check fresh installs, repeat application, and incompatible upstream changes."""

import os
import shutil
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

import pytest

PATCH_DIRECTORY = (
    Path(__file__).resolve().parents[5] / "third_party/alpamayo-transformers5"
)
PACKAGES = (
    ("alpamayo-r1", "alpamayo_r1", "alpamayo-r1.patch"),
    ("alpamayo1-5", "alpamayo1_5", "alpamayo1.5.patch"),
    ("alpamayo2-super", "alpamayo2_super", "alpamayo2.patch"),
)


@pytest.fixture
def unpatched_install(tmp_path):
    site = tmp_path / "site-packages"
    site.mkdir()
    for package, module, filename in PACKAGES:
        installed = distribution(package)
        shutil.copytree(
            installed.locate_file(module),
            site / module,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        metadata = site / f"{module}-0.1.0.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(f"Name: {package}\nVersion: 0.1.0\n")
        (metadata / "direct_url.json").write_text(
            installed.read_text("direct_url.json")
        )
        subprocess.run(
            [
                "git",
                "apply",
                "--unsafe-paths",
                f"--directory={site}",
                "-p2",
                "--reverse",
                str(PATCH_DIRECTORY / filename),
            ],
            check=True,
        )
    return site


def _apply(site):
    return subprocess.run(
        [sys.executable, str(PATCH_DIRECTORY / "apply_patches.py")],
        env={**os.environ, "PYTHONPATH": str(site)},
        capture_output=True,
        text=True,
    )


def test_apply_to_fresh_install_then_repeat(unpatched_install):
    first = _apply(unpatched_install)
    assert first.returncode == 0, first.stderr
    assert first.stdout.count(": patched") == 3
    second = _apply(unpatched_install)
    assert second.returncode == 0, second.stderr
    assert second.stdout.count(": already applied") == 3


def test_incompatible_upstream_fails_before_changing_any_package(unpatched_install):
    first = unpatched_install / "alpamayo_r1/models/base_model.py"
    original = first.read_bytes()
    (unpatched_install / "alpamayo2_super/models/expert.py").write_text(
        "changed upstream\n"
    )
    result = _apply(unpatched_install)
    assert result.returncode != 0
    assert "does not match installed alpamayo2-super" in result.stderr
    assert first.read_bytes() == original


@pytest.mark.parametrize("module", [module for _, module, _ in PACKAGES])
def test_read_only_prompt_cache_layer_works_with_transformers_cache(module):
    """The CUDA-graph expert builds a Cache from this layer on every inference."""
    import importlib

    import torch
    from transformers.cache_utils import Cache

    layer_class = importlib.import_module(
        f"{module}.models.diffusion_expert_cuda_graph"
    )._ReadOnlyPromptCacheLayer
    prompt_keys, prompt_values = torch.zeros(1, 2, 5, 4), torch.ones(1, 2, 5, 4)
    cache = Cache(layers=[layer_class(prompt_keys, prompt_values)])

    assert cache.get_seq_length() == 5
    assert cache.get_max_length() == -1
    assert cache.get_mask_sizes(3, 0) == (8, 0)
    keys, values = cache.update(torch.zeros(1, 2, 3, 4), torch.ones(1, 2, 3, 4), 0)
    assert keys.shape[-2] == values.shape[-2] == 8
    assert cache.get_seq_length() == 5
