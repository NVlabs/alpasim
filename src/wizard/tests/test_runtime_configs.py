# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from pathlib import Path

import alpasim_wizard.setup_omegaconf  # noqa: F401
import pytest
from alpasim_wizard.setup_omegaconf import validate_config
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


def _compose_config(*overrides: str):
    config_dir = Path(__file__).parents[1] / "configs"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name="base_config.yaml", overrides=list(overrides))


def test_usdz_artifact_cache_size_is_only_nested_under_data_source_provider() -> None:
    cfg = _compose_config(
        "deploy=local",
        "topology=8gpu_128rollouts",
        "driver=vavam",
        "wizard.log_dir=/tmp/alpasim-test",
    )

    assert "artifact_cache_size" not in cfg.runtime
    assert cfg.runtime.data_source_provider.usdz.artifact_cache_size == 10


@pytest.mark.parametrize(
    "mps_pipe_dir",
    ["/tmp/example/mps pipes", "/dev/shm/mps;rm", "relative/mps", "/a,b"],
)
def test_rejects_mps_pipe_dir_unsafe_for_shell(mps_pipe_dir: str) -> None:
    cfg = _compose_config(
        "deploy=local",
        "topology=1gpu",
        "driver=vavam",
        "wizard.log_dir=/tmp/alpasim-test",
        "wizard.run_method=SLURM",
        "wizard.enable_mps=true",
    )
    cfg.wizard.mps_pipe_dir = "/dev/shm/alpagym-mps-123"
    validate_config(cfg)

    cfg.wizard.mps_pipe_dir = mps_pipe_dir
    with pytest.raises(ValueError, match="wizard.mps_pipe_dir must be"):
        validate_config(cfg)
