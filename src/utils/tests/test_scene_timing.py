# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import pytest
from alpasim_utils.scene_timing import (
    SceneTimingRejectionReason,
    derive_clip_start_offset_us,
    evaluate_scene_timing,
)
from alpasim_utils.types import T0_AUTO


def test_auto_timing_does_not_require_a_clip_boundary() -> None:
    result = evaluate_scene_timing(
        policy_start_us=T0_AUTO,
        force_gt_duration_us=1_600_000,
        n_sim_steps=38,
        control_timestep_us=200_000,
        clip_duration_us=0,
    )

    assert result.accepted
    assert result.clip_start_offset_us is None
    assert result.requested_end_us is None


def test_explicit_timing_rejects_insufficient_history() -> None:
    result = evaluate_scene_timing(
        policy_start_us=1_599_999,
        force_gt_duration_us=1_600_000,
        n_sim_steps=38,
        control_timestep_us=200_000,
        clip_duration_us=20_000_000,
    )

    assert result.rejection_reason == SceneTimingRejectionReason.INSUFFICIENT_HISTORY
    assert result.requested_end_us is None


def test_explicit_timing_accepts_horizon_at_clip_boundary() -> None:
    result = evaluate_scene_timing(
        policy_start_us=14_000_000,
        force_gt_duration_us=1_600_000,
        n_sim_steps=38,
        control_timestep_us=200_000,
        clip_duration_us=20_000_000,
    )

    assert result.accepted
    assert result.clip_start_offset_us == 12_400_000
    assert result.requested_end_us == 20_000_000


def test_explicit_timing_rejects_horizon_after_clip_boundary() -> None:
    result = evaluate_scene_timing(
        policy_start_us=14_000_000,
        force_gt_duration_us=1_600_000,
        n_sim_steps=38,
        control_timestep_us=200_000,
        clip_duration_us=19_733_638,
    )

    assert result.rejection_reason == SceneTimingRejectionReason.INSUFFICIENT_HORIZON
    assert result.requested_end_us == 20_000_000


def test_derive_clip_start_offset_preserves_existing_history_error() -> None:
    assert derive_clip_start_offset_us(T0_AUTO, 1_600_000) is None
    assert derive_clip_start_offset_us(1_600_000, 1_600_000) == 0
    with pytest.raises(ValueError, match="does not provide"):
        derive_clip_start_offset_us(1_599_999, 1_600_000)
