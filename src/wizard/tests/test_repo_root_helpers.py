# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import alpasim_wizard.setup_omegaconf as setup_omegaconf
import alpasim_wizard.utils as wizard_utils


def test_setup_omegaconf_uses_shared_find_repo_root() -> None:
    assert setup_omegaconf.find_repo_root.__module__ == "alpasim_utils.paths"


def test_wizard_utils_does_not_define_find_repo_root() -> None:
    assert "find_repo_root" not in vars(wizard_utils)
