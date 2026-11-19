# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import pytest
from benchmark.ego_motion_catalog import (
    EgoMotionCatalog,
    make_ego_motion_record,
    record_to_reference_trajectory,
    select_k_disk_cover,
)


def test_record_to_reference_trajectory_rebases_and_resamples() -> None:
    record = make_ego_motion_record(
        name="curve",
        description="curved ego motion",
        timestamps_us=[1_000_000, 1_200_000, 1_400_000],
        x=[0.0, 1.0, 2.0],
        y=[0.0, 0.2, 0.8],
        yaw=[0.0, 0.2, 0.4],
    )

    trajectory = record_to_reference_trajectory(record)

    assert trajectory.name == "curve"
    assert trajectory.points[0].timestamp_us == 0
    assert trajectory.points[1].timestamp_us == 100_000
    assert trajectory.points[-1].timestamp_us == 400_000
    assert trajectory.points[-1].x == pytest.approx(2.0)
    assert trajectory.simulation_duration_us == 400_000


def test_record_to_reference_trajectory_keeps_tail_for_lookahead() -> None:
    timestamps_us = list(range(0, 25_000_001, 1_000_000))
    record = make_ego_motion_record(
        name="long",
        description="25s ego motion",
        timestamps_us=timestamps_us,
        x=[float(i) for i in range(len(timestamps_us))],
        y=[0.0 for _ in timestamps_us],
        yaw=[0.0 for _ in timestamps_us],
    )

    trajectory = record_to_reference_trajectory(record)

    assert trajectory.trajectory_duration_us == 25_000_000
    assert trajectory.simulation_duration_us == 20_000_000


def test_select_k_disk_cover_picks_diverse_ego_motion() -> None:
    records = [
        make_ego_motion_record(
            name="straight_slow",
            description="",
            timestamps_us=[0, 1_000_000, 2_000_000],
            x=[0.0, 2.0, 4.0],
            y=[0.0, 0.0, 0.0],
            yaw=[0.0, 0.0, 0.0],
        ),
        make_ego_motion_record(
            name="straight_fast",
            description="",
            timestamps_us=[0, 1_000_000, 2_000_000],
            x=[0.0, 20.0, 40.0],
            y=[0.0, 0.0, 0.0],
            yaw=[0.0, 0.0, 0.0],
        ),
        make_ego_motion_record(
            name="left_turn",
            description="",
            timestamps_us=[0, 1_000_000, 2_000_000],
            x=[0.0, 8.0, 12.0],
            y=[0.0, 2.0, 8.0],
            yaw=[0.0, 0.25, 0.7],
        ),
    ]

    selected_indices, metadata = select_k_disk_cover(records, k=2)
    selected_names = {records[index].name for index in selected_indices}

    assert len(selected_indices) == 2
    assert len(selected_names) == 2
    assert metadata["method"] == "greedy_k_disk"
    assert metadata["covering_radius"] >= 0.0

    duplicate_records = [
        make_ego_motion_record(
            name="duplicate_a",
            description="",
            timestamps_us=[0, 1_000_000, 2_000_000],
            x=[0.0, 2.0, 4.0],
            y=[0.0, 0.0, 0.0],
            yaw=[0.0, 0.0, 0.0],
        ),
        make_ego_motion_record(
            name="duplicate_b",
            description="",
            timestamps_us=[0, 1_000_000, 2_000_000],
            x=[0.0, 2.0, 4.0],
            y=[0.0, 0.0, 0.0],
            yaw=[0.0, 0.0, 0.0],
        ),
        make_ego_motion_record(
            name="turn",
            description="",
            timestamps_us=[0, 1_000_000, 2_000_000],
            x=[0.0, 8.0, 12.0],
            y=[0.0, 2.0, 8.0],
            yaw=[0.0, 0.25, 0.7],
        ),
    ]

    duplicate_selected_indices, _ = select_k_disk_cover(duplicate_records, k=3)
    duplicate_selected_names = {
        duplicate_records[index].name for index in duplicate_selected_indices
    }

    assert len(duplicate_selected_indices) == 3
    assert len(duplicate_selected_names) == 3


def test_catalog_roundtrip_preserves_selected_records(tmp_path) -> None:
    records = [
        make_ego_motion_record(
            name="a",
            description="",
            timestamps_us=[0, 100_000],
            x=[0.0, 1.0],
            y=[0.0, 0.0],
            yaw=[0.0, 0.0],
        ),
        make_ego_motion_record(
            name="b",
            description="",
            timestamps_us=[0, 100_000],
            x=[0.0, 0.0],
            y=[0.0, 1.0],
            yaw=[1.57, 1.57],
        ),
    ]
    catalog = EgoMotionCatalog(records=records, selected_names=["b"])
    path = tmp_path / "ego_motion_catalog.json"

    catalog.save(path)
    loaded = EgoMotionCatalog.load(path)
    trajectories = loaded.to_reference_trajectories()

    assert loaded.selected_names == ["b"]
    assert [trajectory.name for trajectory in trajectories] == ["b"]
