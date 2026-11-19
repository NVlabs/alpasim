# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Position-based navigation lookup and route-index resolution."""

from __future__ import annotations

import copy
import json
import math
import pickle
import sys
import types
import zipfile

import numpy as np
import pytest
from alpasim_runtime.nav_route import (
    AnchoredTrace,
    NavRouteLibrary,
    NavStatus,
    RouteInstruction,
    RouteNavigator,
    SessionRouteIndex,
    _recent_heading_change_deg,
    load_route_index,
    load_route_index_for_usdz,
)
from alpasim_utils.geometry import Pose, Trajectory

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
SPEED_MPS = 10.0


def _straight_trajectory(timestamps_us: list[int], *, y: float = 0.0) -> Trajectory:
    """Ego driving along +x at SPEED_MPS."""
    return Trajectory.from_poses(
        timestamps=np.array(timestamps_us, dtype=np.uint64),
        poses=[
            Pose(
                np.array([SPEED_MPS * t / 1e6, y, 0.0], dtype=np.float32),
                IDENTITY_QUAT,
            )
            for t in timestamps_us
        ],
    )


def _route_index(
    instructions: list[RouteInstruction] | None = None,
    *,
    session_offset_m: float = 1000.0,
    n_samples: int = 11,
    **overrides,
) -> SessionRouteIndex:
    """Session trace at 1 Hz whose clip starts ``session_offset_m`` into the session."""
    timestamps = np.arange(n_samples, dtype=np.int64) * 1_000_000
    distances = session_offset_m + SPEED_MPS * timestamps / 1e6
    fields = dict(
        timestamps=timestamps,
        cumulative_distances=distances,
        instructions=instructions or [],
        active_maneuver_lookback_m=10.0,
        max_instruction_distance_m=250.0,
    )
    fields.update(overrides)
    return SessionRouteIndex(**fields)


def _turn_left_at(session_distance_m: float) -> RouteInstruction:
    return RouteInstruction(session_distance_m, "TURN_LEFT", "Turn left onto Main St")


@pytest.fixture
def navigator() -> RouteNavigator:
    """Clip covering session distances 1000..1100 m with a left turn at 1080 m."""
    return RouteNavigator(
        _route_index([_turn_left_at(1080.0)]),
        _straight_trajectory(list(range(0, 10_000_001, 100_000))),
    )


def test_instruction_distance_follows_the_ego_position(navigator: RouteNavigator):
    """An ego 30 m into the clip is 50 m from the turn whatever the clock says."""
    instruction = navigator.lookup(np.array([30.0, 0.0, 0.0]))

    assert instruction.status is NavStatus.UPCOMING
    assert instruction.maneuver == "TURN_LEFT"
    assert instruction.instruction_text == "Turn left onto Main St in 50m"
    assert instruction.remaining_distance_m == pytest.approx(50.0)
    assert instruction.route_offset_m == pytest.approx(0.0)


def test_lateral_deviation_is_reported_but_does_not_move_the_ego_along_the_route(
    navigator: RouteNavigator,
):
    instruction = navigator.lookup(np.array([30.0, 2.5, 0.0]))

    assert instruction.remaining_distance_m == pytest.approx(50.0)
    assert instruction.route_offset_m == pytest.approx(2.5)


def test_a_maneuver_stays_active_while_the_ego_clears_it(navigator: RouteNavigator):
    instruction = navigator.lookup(np.array([85.0, 0.0, 0.0]))

    assert instruction.status is NavStatus.ACTIVE
    assert instruction.instruction_text == "Turn left onto Main St"
    assert instruction.remaining_distance_m == 0.0


def test_route_is_complete_after_the_last_maneuver_and_its_lookback(
    navigator: RouteNavigator,
):
    instruction = navigator.lookup(np.array([95.0, 0.0, 0.0]))

    assert instruction.status is NavStatus.ROUTE_COMPLETE
    assert instruction.maneuver == "STRAIGHT"
    assert instruction.instruction_text == "Continue straight"


def test_a_maneuver_beyond_the_horizon_reads_as_continue_straight():
    navigator = RouteNavigator(
        _route_index([_turn_left_at(1300.0)]),
        _straight_trajectory([0, 5_000_000, 10_000_000]),
    )

    instruction = navigator.lookup(np.array([20.0, 0.0, 0.0]))

    assert instruction.status is NavStatus.OUTSIDE_HORIZON
    assert instruction.instruction_text == "Continue straight"


def test_ego_past_the_recording_end_keeps_a_consistent_distance(
    navigator: RouteNavigator,
):
    """The anchored trace ends at 100 m; 120 m along it is 1120 m into the session."""
    navigator = RouteNavigator(
        _route_index([_turn_left_at(1150.0)]),
        _straight_trajectory(list(range(0, 10_000_001, 100_000))),
    )

    cumulative_m, offset_m = navigator.cumulative_distance_at(
        np.array([120.0, 1.0, 0.0])
    )

    assert cumulative_m == pytest.approx(1120.0)
    assert offset_m == pytest.approx(1.0)
    assert navigator.lookup(np.array([120.0, 1.0, 0.0])).remaining_distance_m == (
        pytest.approx(30.0)
    )


def test_ego_before_the_recording_start_extends_the_first_segment(
    navigator: RouteNavigator,
):
    cumulative_m, offset_m = navigator.cumulative_distance_at(
        np.array([-10.0, 0.0, 0.0])
    )

    assert cumulative_m == pytest.approx(990.0)
    assert offset_m == pytest.approx(0.0)


def test_stationary_recording_samples_do_not_break_the_projection():
    """A recording that waits produces repeated anchors with zero-length segments."""
    timestamps = [0, 1_000_000, 2_000_000, 3_000_000]
    positions = [0.0, 10.0, 10.0, 20.0]
    trajectory = Trajectory.from_poses(
        timestamps=np.array(timestamps, dtype=np.uint64),
        poses=[
            Pose(np.array([x, 0.0, 0.0], dtype=np.float32), IDENTITY_QUAT)
            for x in positions
        ],
    )
    route_index = _route_index(
        n_samples=4,
        cumulative_distances=1000.0 + np.array(positions),
    )
    navigator = RouteNavigator(route_index, trajectory)

    cumulative_m, _ = navigator.cumulative_distance_at(np.array([15.0, 0.0, 0.0]))

    assert cumulative_m == pytest.approx(1015.0)


def test_navigator_needs_two_route_samples_inside_the_clip():
    route_index = _route_index(timestamps=np.array([50_000_000, 51_000_000]))

    with pytest.raises(ValueError, match="inside the clip time range"):
        RouteNavigator(route_index, _straight_trajectory([0, 1_000_000]))


def _pickle_as_module(route_index: SessionRouteIndex, module_name: str) -> bytes:
    """Pickle a route index as if its classes lived in ``module_name``."""
    module = types.ModuleType(module_name)
    legacy_classes = {
        cls.__name__: type(cls.__name__, (cls,), {"__module__": module_name})
        for cls in (RouteInstruction, SessionRouteIndex)
    }
    for name, cls in legacy_classes.items():
        setattr(module, name, cls)
    legacy = copy.deepcopy(route_index)
    legacy.__class__ = legacy_classes["SessionRouteIndex"]
    for instruction in legacy.instructions:
        instruction.__class__ = legacy_classes["RouteInstruction"]
    # pickle imports the module and its parents to check the class resolves.
    parents = module_name.split(".")
    stubs = {
        ".".join(parents[: i + 1]): types.ModuleType(".".join(parents[: i + 1]))
        for i in range(len(parents) - 1)
    }
    stubs[module_name] = module
    sys.modules.update(stubs)
    try:
        return pickle.dumps(legacy)
    finally:
        for name in stubs:
            del sys.modules[name]


@pytest.mark.parametrize(
    "module_name", ["route_index", "alpamayo.data.nav_route.route_index"]
)
def test_route_index_pickles_load_from_their_original_module_paths(
    tmp_path, module_name: str
):
    path = tmp_path / "session.pkl"
    path.write_bytes(
        _pickle_as_module(_route_index([_turn_left_at(1080.0)]), module_name)
    )

    loaded = load_route_index(str(path), "clipgt-session")

    assert type(loaded) is SessionRouteIndex
    assert type(loaded.instructions[0]) is RouteInstruction
    assert loaded.instructions[0].instruction_text == "Turn left onto Main St"
    np.testing.assert_array_equal(loaded.timestamps, _route_index().timestamps)


def test_library_loads_clip_routes_across_suites_and_preserves_position_lookup(
    tmp_path,
):
    root = tmp_path / "v3"
    first = root / "public_2601" / "aaa.pkl"
    second = root / "public_2604" / "bbb.pkl"
    index = _route_index([_turn_left_at(1080.0)])
    for path in (first, second):
        path.parent.mkdir(parents=True)
        path.write_bytes(pickle.dumps(index))

    library = NavRouteLibrary.build(
        str(root), {"clipgt-aaa", "clipgt-bbb", "clipgt-missing"}
    )
    trajectory = _straight_trajectory([0, 5_000_000, 10_000_000])
    for clip_id in ("clipgt-aaa", "clipgt-bbb"):
        navigator = RouteNavigator(library.route_index_for_clip(clip_id), trajectory)
        assert navigator.lookup(np.array([30.0, 0.0, 0.0])).instruction_text == (
            "Turn left onto Main St in 50m"
        )
        assert navigator.lookup(np.array([85.0, 0.0, 0.0])).status is NavStatus.ACTIVE
        assert (
            navigator.lookup(np.array([95.0, 0.0, 0.0])).instruction_text
            == "Continue straight"
        )
    with pytest.raises(KeyError):
        library.route_index_for_clip("clipgt-missing")


def test_published_json_route_index_matches_the_pickled_one(tmp_path):
    """A clip's entry in a published JSON file drives the navigator like a pickle."""
    pickled = _route_index([_turn_left_at(1080.0), _turn_left_at(1400.0)])
    (tmp_path / "pickled").mkdir()
    (tmp_path / "pickled" / "aaa.pkl").write_bytes(pickle.dumps(pickled))
    (tmp_path / "public").mkdir()
    published = {
        "timestamps_us": pickled.timestamps.tolist(),
        "cumulative_distances_m": (pickled.cumulative_distances - 1000.0).tolist(),
        "instructions": [
            {
                "cumulative_distance_m": instr.cumulative_distance_m - 1000.0,
                "maneuver": instr.maneuver,
                "instruction_text": instr.instruction_text,
            }
            for instr in pickled.instructions
        ],
        "active_maneuver_lookback_m": 10.0,
        "max_instruction_distance_m": 250.0,
        "roundabout_phase_lookback_m": 0.0,
    }
    (tmp_path / "public" / "route_indices.json").write_text(
        json.dumps({"aaa": published, "bbb": published})
    )
    trajectory = _straight_trajectory(list(range(0, 10_000_001, 100_000)))
    navigators = [
        RouteNavigator(
            NavRouteLibrary.build(
                str(tmp_path / root), {"clipgt-aaa"}
            ).route_index_for_clip("clipgt-aaa"),
            trajectory,
        )
        for root in ("pickled", "public")
    ]

    for x in (0.0, 30.0, 85.0, 95.0, 140.0):
        from_pickle, public = (n.lookup(np.array([x, 0.0, 0.0])) for n in navigators)
        assert public == from_pickle


def _arc_trajectory(heading_rate_deg_per_m: float) -> Trajectory:
    """Ego driving 10 m/s along an arc turning at a constant rate per metre."""
    timestamps = list(range(0, 10_000_001, 100_000))
    heading = [
        math.radians(heading_rate_deg_per_m * SPEED_MPS * t / 1e6) for t in timestamps
    ]
    xy = np.cumsum(
        [[0.0, 0.0]]
        + [
            [math.cos(h) * SPEED_MPS * 0.1, math.sin(h) * SPEED_MPS * 0.1]
            for h in heading[1:]
        ],
        axis=0,
    )
    return Trajectory.from_poses(
        timestamps=np.array(timestamps, dtype=np.uint64),
        poses=[
            Pose(np.array([x, y, 0.0], dtype=np.float32), IDENTITY_QUAT) for x, y in xy
        ],
    )


def _roundabout_navigator(
    heading_rate_deg_per_m: float,
) -> tuple[RouteNavigator, np.ndarray]:
    """Navigator for an ego on an arc, and its position 12 m before a roundabout."""
    trajectory = _arc_trajectory(heading_rate_deg_per_m)
    entry = RouteInstruction(1100.0, "ROUNDABOUT_LEFT", "Enter roundabout onto Road")
    return (
        RouteNavigator(
            _route_index([entry], roundabout_phase_lookback_m=20.0), trajectory
        ),
        trajectory.positions[88, :2],
    )


@pytest.mark.parametrize(
    ("heading_rate_deg_per_m", "expected"),
    [
        (0.0, "Enter roundabout onto Road in 12m"),
        (4.0, "Continue around the roundabout and take the next exit"),
    ],
)
def test_roundabout_phase_follows_the_recorded_trajectory_heading(
    heading_rate_deg_per_m: float, expected: str
):
    """An ego already turning through the roundabout is told to keep going round."""
    navigator, position = _roundabout_navigator(heading_rate_deg_per_m)

    instruction = navigator.lookup(np.array([*position, 0.0]))

    assert instruction.instruction_text == expected


def test_library_build_fails_when_no_configured_clip_has_a_route_index(tmp_path):
    with pytest.raises(ValueError, match="none of the 1 configured clips"):
        NavRouteLibrary.build(str(tmp_path), {"clipgt-aaa"})


@pytest.mark.parametrize("missing", [False, True])
def test_usdz_navigation_lookup_uses_embedded_clip_id(tmp_path, missing):
    usdz = tmp_path / "unrelated-artifact-uuid.usdz"
    with zipfile.ZipFile(usdz, "w") as archive:
        archive.writestr(
            "metadata.yaml",
            "scene_id: clipgt-aaa\nversion_string: test\ntraining_date: test\n"
            "dataset_hash: test\nuuid: artifact-uuid\nis_resumable: false\n"
            "sensors: {}\nlogger: {}\ntime_range: {start: 0, end: 1}\n",
        )
    if not missing:
        suite = tmp_path / "suite"
        suite.mkdir()
        (suite / "aaa.pkl").write_bytes(pickle.dumps(_route_index()))

    if missing:
        with pytest.raises(ValueError, match="none of the 1 configured clips"):
            load_route_index_for_usdz(usdz, str(tmp_path))
    else:
        record = load_route_index_for_usdz(usdz, str(tmp_path))
        np.testing.assert_array_equal(record.timestamps, _route_index().timestamps)


def test_a_waiting_ego_hears_the_plain_roundabout_entry():
    """Position noise while waiting at the entry must not read as turning."""
    rng = np.random.default_rng(0)
    timestamps = list(range(0, 10_000_001, 100_000))
    poses = [
        Pose(
            np.array(
                [min(SPEED_MPS * t / 1e6, 30.0), 0.0, 0.0]
                + np.r_[rng.normal(0.0, 0.01, 2), 0.0] * (t > 3_000_000),
                dtype=np.float32,
            ),
            IDENTITY_QUAT,
        )
        for t in timestamps
    ]
    trajectory = Trajectory.from_poses(
        timestamps=np.array(timestamps, dtype=np.uint64), poses=poses
    )
    entry = RouteInstruction(1042.0, "ROUNDABOUT_LEFT", "Enter roundabout onto Road")
    route_index = _route_index(
        [entry],
        cumulative_distances=1000.0 + np.minimum(np.arange(11) * SPEED_MPS, 30.0),
        roundabout_phase_lookback_m=20.0,
    )
    navigator = RouteNavigator(route_index, trajectory)

    instruction = navigator.lookup(np.array([*trajectory.positions[-1, :2], 0.0]))

    assert instruction.instruction_text == "Enter roundabout onto Road in 12m"


def test_route_progress_without_recorded_motion_gives_no_heading():
    """Route distance that advances while the recorded ego stays put has no heading."""
    rng = np.random.default_rng(0)
    trace = AnchoredTrace(
        xy=rng.normal(0.0, 0.01, (21, 2)), distances_m=np.linspace(0.0, 20.0, 21)
    )

    assert _recent_heading_change_deg(trace, 20.0, 20.0) is None


def test_heading_needs_the_whole_lookback_on_the_recording():
    """12 m into the clip, a 20 m look-back would reach before the recording."""
    trajectory = _arc_trajectory(6.0)
    entry = RouteInstruction(1020.0, "ROUNDABOUT_LEFT", "Enter roundabout onto Road")
    navigator = RouteNavigator(
        _route_index([entry], roundabout_phase_lookback_m=20.0), trajectory
    )

    instruction = navigator.lookup(np.array([*trajectory.positions[12, :2], 0.0]))

    assert instruction.instruction_text == "Enter roundabout onto Road in 8m"
