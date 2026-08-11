# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import alpasim_wizard.setup_omegaconf as setup_omegaconf
import alpasim_wizard.utils as wizard_utils


def test_setup_omegaconf_uses_shared_find_repo_root() -> None:
    assert setup_omegaconf.find_repo_root.__module__ == "alpasim_utils.paths"


def test_wizard_utils_does_not_define_find_repo_root() -> None:
    assert "find_repo_root" not in vars(wizard_utils)


def test_base_image_tag_uses_version_on_main(monkeypatch) -> None:
    monkeypatch.setattr(setup_omegaconf, "_read_repo_version", lambda: "0.115.0")
    monkeypatch.setattr(
        setup_omegaconf,
        "_git_output",
        lambda *args: "main" if args == ("rev-parse", "--abbrev-ref", "HEAD") else None,
    )

    assert setup_omegaconf._read_base_image_tag() == "0.115.0"


def test_base_image_tag_uses_commit_for_mr_base_image_change(monkeypatch) -> None:
    responses = {
        ("rev-parse", "--abbrev-ref", "HEAD"): "feature/prefetch",
        ("merge-base", "HEAD", "origin/main"): "abc123",
        ("diff", "--name-only", "abc123...HEAD"): "pyproject.toml\nsrc/runtime/x.py",
        ("rev-parse", "HEAD"): "cf8241f7ee050451475cd698898d0e451f9c4a99",
    }
    monkeypatch.setattr(setup_omegaconf, "_read_repo_version", lambda: "0.115.0")
    monkeypatch.setattr(setup_omegaconf, "_git_output", lambda *args: responses[args])

    assert setup_omegaconf._read_base_image_tag() == "cf8241f7"


def test_base_image_tag_uses_version_for_feature_without_base_image_change(
    monkeypatch,
) -> None:
    responses = {
        ("rev-parse", "--abbrev-ref", "HEAD"): "feature/config",
        ("merge-base", "HEAD", "origin/main"): "abc123",
        ("diff", "--name-only", "abc123...HEAD"): "src/wizard/configs/base_config.yaml",
    }
    monkeypatch.setattr(setup_omegaconf, "_read_repo_version", lambda: "0.115.0")
    monkeypatch.setattr(setup_omegaconf, "_git_output", lambda *args: responses[args])

    assert setup_omegaconf._read_base_image_tag() == "0.115.0"
