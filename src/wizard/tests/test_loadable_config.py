# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import tempfile

import pytest
from alpasim_wizard.utils import (
    _process_config_values_for_saving,
    nre_image_to_nre_version,
)
from omegaconf import OmegaConf


class TestLoadableConfig:
    """Test config processing for save/load round-trip with escaped interpolations."""

    def test_save_and_load_round_trip_preserves_literal_strings(self):
        """Test that saving and loading configs preserves literal strings with ${}."""
        # Create config with strings that look like interpolations but should be literal
        cfg = {
            "literal1": "This is a literal: ${not_an_interpolation}",
            "literal2": "This is a literal: \\${not_an_interpolation}",
            "literal3": "This is a literal: \\\\${not_an_interpolation}",
            "literal4": "This is a literal: \\\\\\${not_an_interpolation}",
        }

        # Process config for saving (escape $ characters)
        cfg_escaped = OmegaConf.create(_process_config_values_for_saving(cfg))

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as tmp_file:
            save_path = tmp_file.name
            OmegaConf.save(cfg_escaped, save_path)
            loaded_cfg = OmegaConf.load(save_path)

            # Verify the loaded values are exactly what we expect
            assert loaded_cfg.literal1 == cfg["literal1"]
            assert loaded_cfg.literal2 == cfg["literal2"]
            assert loaded_cfg.literal3 == cfg["literal3"]
            assert loaded_cfg.literal4 == cfg["literal4"]


def test_nre_image_to_nre_version_supports_legacy_and_http_formats():
    assert (
        nre_image_to_nre_version("nvcr.io/nvidian/alpamayo/nre:25.7.9-dc9a8043")
        == "25.7.9-dc9a8043"
    )
    assert (
        nre_image_to_nre_version(
            "http://nvcr.io/nvidian/ct-toronto-ai/nre_run:26.3.11-d7f37b9c-dev "
        )
        == "26.3.11-d7f37b9c-dev"
    )


def test_nre_image_to_nre_version_rejects_references_without_tag():
    with pytest.raises(ValueError):
        nre_image_to_nre_version("nvcr.io/nvidian/ct-toronto-ai/nre_run")
