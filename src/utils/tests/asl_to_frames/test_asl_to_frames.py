# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

from alpasim_utils.asl_to_frames.__main__ import determine_save_dir


def test_determine_save_dir():
    log_path = "mnt/rollouts/cliggt-hash/rollout.asl"

    # nominal path, log_save_dir unspecified
    save_dir = determine_save_dir(log_path, None)
    expected_save_dir = "mnt/rollouts/cliggt-hash/rollout_asl_frames"
    assert save_dir == expected_save_dir, f"{save_dir=} {expected_save_dir=}"

    # path required by kpi, log_save_dir specified
    save_dir = determine_save_dir(log_path, "mnt/outputs")
    expected_save_dir = "mnt/outputs/rollouts/cliggt-hash/rollout"
    assert save_dir == expected_save_dir, f"{save_dir=} {expected_save_dir=}"
