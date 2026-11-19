# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import logging

import pytest
from alpasim_wizard.nav_coverage import check_recorded_navigation_coverage


def test_wizard_warns_about_every_scene_without_recorded_navigation(tmp_path, caplog):
    route_root = tmp_path / "route"
    route_dir = route_root / "public_2601"
    route_dir.mkdir(parents=True)
    (route_dir / "aaa.pkl").touch()
    other_suite = route_root / "public_2604"
    other_suite.mkdir()
    (other_suite / "bbb.pkl").touch()
    simulation_config = {
        "nav_instruction": {
            "source": "recorded",
            "route_index_dir": "/mnt/route",
        }
    }

    with caplog.at_level(logging.WARNING, logger="alpasim_wizard.nav_coverage"):
        check_recorded_navigation_coverage(
            simulation_config,
            {"clipgt-aaa", "clipgt-bbb", "clipgt-ccc", "clipgt-ddd"},
            [f"{route_root}:/mnt/route:ro"],
        )

    assert caplog.messages == [
        "Recorded navigation is available for 2/4 configured clips. "
        "The following 2 clips have no navigation input and will fail during "
        "rollout setup:\n"
        "- clipgt-ccc\n"
        "- clipgt-ddd"
    ]


def test_wizard_fails_when_no_requested_scene_has_recorded_navigation(tmp_path, caplog):
    route_root = tmp_path / "route"
    (route_root / "unrelated-suite").mkdir(parents=True)
    (route_root / "unrelated-suite" / "other-clip.pkl").touch()
    simulation_config = {
        "nav_instruction": {
            "source": "recorded",
            "route_index_dir": "/mnt/route",
        }
    }

    with caplog.at_level(logging.WARNING, logger="alpasim_wizard.nav_coverage"):
        with pytest.raises(ValueError, match="none of the 2 configured clips"):
            check_recorded_navigation_coverage(
                simulation_config,
                {"clipgt-aaa", "clipgt-bbb"},
                [f"{route_root}:/mnt/route:ro"],
            )

    assert caplog.messages == [
        "Recorded navigation is available for 0/2 configured clips. "
        "The following 2 clips have no navigation input and will fail during "
        "rollout setup:\n"
        "- clipgt-aaa\n"
        "- clipgt-bbb"
    ]
