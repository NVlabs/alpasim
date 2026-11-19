# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Turn-by-turn navigation from pre-computed route indices.

A route index (``SessionRouteIndex``) holds a recording's route trace (timestamps and
cumulative distance) and the routed maneuvers placed along that cumulative distance.
Route indices are either ``<bare_clip_id>.pkl`` files beneath a navigation version
root, grouped by scene suite, that cover the whole recording session, or published
``route_indices*.json`` files mapping bare clip IDs to route indices that cover only the clip,
without GPS (see ``data/nav``).

In closed loop the simulated ego leaves the recorded trace, so instructions are
looked up by position rather than by the recording's clock. The recorded ego
trajectory anchors the route's cumulative distance in the scene's local
frame; the simulated ego is projected onto the anchored trace and the maneuvers
are read off at the projected cumulative distance. An ego that is ahead of or
behind the recording sees the same turn approach at the right distance.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import pickle
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from alpasim_utils.artifact import Artifact
from alpasim_utils.geometry import Trajectory
from alpasim_utils.nav_route import find_route_indices
from alpasim_utils.nav_text import route_to_nav_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Route index
# ---------------------------------------------------------------------------


@dataclass
class RouteInstruction:
    """A single navigation instruction at a point along the route."""

    cumulative_distance_m: float
    maneuver: str  # e.g. "TURN_LEFT", "TURN_RIGHT"
    instruction_text: str  # e.g. "Turn left onto Street", without the distance


@dataclass
class SessionRouteIndex:
    """Route trace and maneuvers along a recording."""

    timestamps: np.ndarray  # (N,) int64 microseconds, same clock as the clip
    cumulative_distances: np.ndarray  # (N,) float64 meters

    instructions: list[RouteInstruction] = field(default_factory=list)

    # A maneuver stays reported while the ego is within this distance past it.
    active_maneuver_lookback_m: float = 0.0
    # Maneuvers further ahead than this are not reported ("Continue straight").
    max_instruction_distance_m: float | None = None
    # Route distance over which an imminent roundabout entry checks whether the ego
    # is already turning through it.
    roundabout_phase_lookback_m: float = 0.0


class _PickledRouteIndex:
    """Attributes of a pickled session route index.

    V3 pickles predate the lookup settings and take these defaults.
    """

    active_maneuver_lookback_m = 0.0
    max_instruction_distance_m = None
    roundabout_phase_lookback_m = 0.0


class _PickledRouteInstruction:
    """Attributes of a pickled route instruction."""


class _RouteIndexUnpickler(pickle.Unpickler):
    """Read route-index pickles regardless of the module that wrote them."""

    _PICKLED_MODULES = frozenset(
        {"route_index", "alpamayo.data.nav_route.route_index", __name__}
    )

    def find_class(self, module: str, name: str):
        if module in self._PICKLED_MODULES:
            return {
                "RouteInstruction": _PickledRouteInstruction,
                "SessionRouteIndex": _PickledRouteIndex,
            }[name]
        return super().find_class(module, name)


def _route_index_from_pickle(path: str) -> SessionRouteIndex:
    """Route index of a session pickle; its GPS and other fields are unused."""
    with open(path, "rb") as f:
        raw = _RouteIndexUnpickler(f).load()
    if not isinstance(raw, _PickledRouteIndex):
        raise TypeError(f"{path} does not contain a SessionRouteIndex")
    return SessionRouteIndex(
        timestamps=np.asarray(raw.timestamps, dtype=np.int64),
        cumulative_distances=np.asarray(raw.cumulative_distances, dtype=np.float64),
        instructions=[
            RouteInstruction(
                cumulative_distance_m=instr.cumulative_distance_m,
                maneuver=instr.maneuver,
                instruction_text=instr.instruction_text,
            )
            for instr in raw.instructions
        ],
        active_maneuver_lookback_m=raw.active_maneuver_lookback_m,
        max_instruction_distance_m=raw.max_instruction_distance_m,
        roundabout_phase_lookback_m=raw.roundabout_phase_lookback_m,
    )


def _route_index_from_json(data: dict) -> SessionRouteIndex:
    """Route index of one clip's entry in a published JSON file (format in ``data/nav``)."""
    return SessionRouteIndex(
        timestamps=np.asarray(data["timestamps_us"], dtype=np.int64),
        cumulative_distances=np.asarray(
            data["cumulative_distances_m"], dtype=np.float64
        ),
        instructions=[RouteInstruction(**instr) for instr in data["instructions"]],
        active_maneuver_lookback_m=data["active_maneuver_lookback_m"],
        max_instruction_distance_m=data["max_instruction_distance_m"],
        roundabout_phase_lookback_m=data["roundabout_phase_lookback_m"],
    )


@functools.lru_cache(maxsize=8)
def _published_route_indices(path: str) -> dict[str, dict]:
    """A published JSON file mapping bare clip IDs to route indices, parsed once."""
    return json.loads(Path(path).read_text())


@functools.lru_cache(maxsize=128)
def _load_pickled_route_index(path: str) -> SessionRouteIndex:
    return _route_index_from_pickle(path)


def load_route_index(path: str, clip_id: str) -> SessionRouteIndex:
    """Route index of a clip from its ``.pkl`` or a published ``.json`` file."""
    if path.endswith(".json"):
        bare = clip_id.removeprefix("clipgt-")
        return _route_index_from_json(_published_route_indices(path)[bare])
    return _load_pickled_route_index(path)


# ---------------------------------------------------------------------------
# Instruction at a cumulative distance
# ---------------------------------------------------------------------------


class NavStatus(Enum):
    UPCOMING = "upcoming"
    ACTIVE = "active"
    OUTSIDE_HORIZON = "outside_horizon"
    ROUTE_COMPLETE = "route_complete"
    # No instruction applies at this step (e.g. the generated route has no usable
    # waypoint); the driver drops any earlier instruction.
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class NavInstruction:
    """Navigation state reported to the driver for one policy step."""

    maneuver: str
    instruction_text: str
    remaining_distance_m: float
    status: NavStatus
    # Lateral distance from the ego to the recorded route.
    route_offset_m: float


# Sent when a navigator has nothing to say this step, so a stale instruction does not
# linger in the driver's session.
NO_INSTRUCTION = NavInstruction(
    maneuver="",
    instruction_text="",
    remaining_distance_m=0.0,
    status=NavStatus.UNAVAILABLE,
    route_offset_m=0.0,
)


def _is_actionable_maneuver(maneuver: str) -> bool:
    """Return whether an instruction describes a localized driving decision."""
    return maneuver.startswith(("TURN_", "U_TURN_", "ROUNDABOUT_", "RAMP_", "FORK_"))


# Each half of the heading look-back must move the recorded ego this far in a straight
# line; below it, position noise of a stopped or creeping ego would dominate.
MIN_HEADING_CHORD_M = 0.5


def _recent_heading_change_deg(
    trace: AnchoredTrace, ego_cumulative_distance_m: float, lookback_m: float
) -> float | None:
    """Heading change between the two halves of the recorded route behind the ego.

    The route positions at the start, middle and end of the look-back are interpolated
    along the recorded trajectory. None when the recording does not cover the whole
    look-back or either half moved less than ``MIN_HEADING_CHORD_M``.
    """
    distances = trace.distances_m
    start_m = ego_cumulative_distance_m - lookback_m
    if start_m < distances[0] or ego_cumulative_distance_m > distances[-1]:
        return None
    at = [
        start_m,
        (start_m + ego_cumulative_distance_m) / 2.0,
        ego_cumulative_distance_m,
    ]
    points = np.stack(
        [
            np.interp(at, distances, trace.xy[:, 0]),
            np.interp(at, distances, trace.xy[:, 1]),
        ],
        axis=1,
    )
    first, second = points[1] - points[0], points[2] - points[1]
    if min(np.linalg.norm(first), np.linalg.norm(second)) < MIN_HEADING_CHORD_M:
        return None
    change = math.degrees(
        math.atan2(second[1], second[0]) - math.atan2(first[1], first[0])
    )
    return (change + 180.0) % 360.0 - 180.0


def _should_continue_inside_roundabout(
    route_index: SessionRouteIndex,
    trace: AnchoredTrace,
    instruction: RouteInstruction,
    ego_cumulative_distance_m: float,
    remaining_distance_m: float,
) -> bool:
    """Return whether an imminent roundabout instruction is already in progress."""
    lookback_m = float(route_index.roundabout_phase_lookback_m)
    if (
        lookback_m <= 0.0
        or not instruction.maneuver.startswith("ROUNDABOUT_")
        or not instruction.instruction_text.startswith("Enter roundabout")
        or remaining_distance_m > 15.0
    ):
        return False
    heading_change = _recent_heading_change_deg(
        trace, ego_cumulative_distance_m, lookback_m
    )
    return heading_change is not None and abs(heading_change) >= 30.0


@dataclass(frozen=True)
class AnchoredTrace:
    """The recorded trajectory in the scene's local frame with route distances."""

    xy: np.ndarray  # (M, 2) local-frame positions
    distances_m: np.ndarray  # (M,) non-decreasing route cumulative distances


def instruction_at_distance(
    route_index: SessionRouteIndex,
    trace: AnchoredTrace,
    ego_cumulative_distance_m: float,
    route_offset_m: float = 0.0,
) -> NavInstruction:
    """Report the navigation instruction for an ego at a cumulative route distance.

    A maneuver is *upcoming* while it lies ahead within
    ``max_instruction_distance_m``, *active* while the ego is within
    ``active_maneuver_lookback_m`` past it, and otherwise the ego is told to
    continue straight, either because the next decision is beyond the horizon or
    because the route holds no further maneuvers. This mirrors the training-time
    lookup so the driver sees the wording the model was trained on.
    """
    maneuver = "STRAIGHT"
    text = "Continue straight"
    remaining_m = 0.0
    status = NavStatus.ROUTE_COMPLETE

    active: RouteInstruction | None = None
    lookback_m = float(route_index.active_maneuver_lookback_m)
    horizon_m = route_index.max_instruction_distance_m
    for instr in route_index.instructions:
        remaining = instr.cumulative_distance_m - ego_cumulative_distance_m
        if -lookback_m <= remaining <= 0.0 and _is_actionable_maneuver(instr.maneuver):
            active = instr
        if remaining > 0:
            if active is None and (horizon_m is None or remaining <= float(horizon_m)):
                status = NavStatus.UPCOMING
                maneuver = instr.maneuver
                if _should_continue_inside_roundabout(
                    route_index, trace, instr, ego_cumulative_distance_m, remaining
                ):
                    text = "Continue around the roundabout and take the next exit"
                else:
                    text = f"{instr.instruction_text} in {remaining:.0f}m"
                remaining_m = remaining
            elif active is None:
                status = NavStatus.OUTSIDE_HORIZON
            break
    if active is not None:
        status = NavStatus.ACTIVE
        maneuver = active.maneuver
        text = active.instruction_text
        remaining_m = 0.0

    return NavInstruction(
        maneuver=maneuver,
        instruction_text=text,
        remaining_distance_m=float(remaining_m),
        status=status,
        route_offset_m=float(route_offset_m),
    )


# ---------------------------------------------------------------------------
# Position-based lookup for one rollout
# ---------------------------------------------------------------------------


class RouteNavigator:
    """Look up navigation instructions from the ego's position in the local frame.

    The route index's samples that fall inside the clip anchor the route's
    cumulative distance to the recorded ego trajectory: at each sample time the
    recorded position carries that sample's distance. The recording's own poses
    between anchors are added with distances interpolated by arc length, so the
    anchored trace follows curves instead of cutting chords between 1 Hz route
    samples. Both the clip and the route index use the recording's clock.
    """

    def __init__(
        self, route_index: SessionRouteIndex, recorded_ego_trajectory: Trajectory
    ) -> None:
        gps_timestamps = np.asarray(route_index.timestamps, dtype=np.int64)
        clip_range = recorded_ego_trajectory.time_range_us
        inside = (gps_timestamps >= clip_range.start) & (
            gps_timestamps < clip_range.stop
        )
        if inside.sum() < 2:
            raise ValueError(
                f"route index has {int(inside.sum())} samples inside the clip time range "
                f"[{clip_range.start}, {clip_range.stop}); at least 2 are needed"
            )
        gps_timestamps = gps_timestamps[inside]
        gps_distances = np.asarray(route_index.cumulative_distances, dtype=np.float64)[
            inside
        ]

        recorded_timestamps = recorded_ego_trajectory.timestamps_us.astype(np.int64)
        between = (recorded_timestamps >= gps_timestamps[0]) & (
            recorded_timestamps <= gps_timestamps[-1]
        )
        timestamps = np.union1d(gps_timestamps, recorded_timestamps[between])
        trace = recorded_ego_trajectory.interpolate(timestamps.astype(np.uint64))
        trace_xy = trace.positions[:, :2].astype(np.float64)
        arc_length = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(trace_xy, axis=0), axis=1)))
        )
        arc_length_at_gps = arc_length[np.searchsorted(timestamps, gps_timestamps)]

        self._route_index = route_index
        self._trace = AnchoredTrace(
            xy=trace_xy,
            distances_m=np.interp(arc_length, arc_length_at_gps, gps_distances),
        )

    def cumulative_distance_at(self, position_local: np.ndarray) -> tuple[float, float]:
        """Project a local-frame position onto the anchored route.

        Returns the session cumulative distance at the projection and the lateral
        offset from the route. Beyond either end of the clip the end segment is
        extended, so an ego that overruns the recording keeps a consistent distance.
        """
        point = np.asarray(position_local, dtype=np.float64)[:2]
        starts = self._trace.xy[:-1]
        segments = np.diff(self._trace.xy, axis=0)
        lengths_sq = np.einsum("ij,ij->i", segments, segments)
        rel = point - starts
        # Stationary recording samples produce zero-length segments; their
        # projection is the sample itself.
        params = np.divide(
            np.einsum("ij,ij->i", rel, segments),
            lengths_sq,
            out=np.zeros_like(lengths_sq),
            where=lengths_sq > 0.0,
        )
        clamped = np.clip(params, 0.0, 1.0)
        projections = starts + clamped[:, None] * segments
        offsets = np.linalg.norm(projections - point, axis=1)
        best = int(np.argmin(offsets))

        param = clamped[best]
        offset_m = offsets[best]
        last = len(segments) - 1
        extends_start = best == 0 and params[0] < 0.0
        extends_end = best == last and params[last] > 1.0
        if (extends_start or extends_end) and lengths_sq[best] > 0.0:
            param = params[best]
            direction = segments[best] / math.sqrt(lengths_sq[best])
            offset_m = abs(
                float(direction[0] * rel[best][1] - direction[1] * rel[best][0])
            )

        distance_span = (
            self._trace.distances_m[best + 1] - self._trace.distances_m[best]
        )
        cumulative_m = self._trace.distances_m[best] + param * distance_span
        return float(cumulative_m), float(offset_m)

    def lookup(
        self, position_local: np.ndarray, route_in_rig: np.ndarray | None = None
    ) -> NavInstruction:
        """Navigation instruction for an ego at a local-frame position.

        ``route_in_rig`` is accepted so both navigators share one call; the
        recorded route index does not need the generated route.
        """
        cumulative_m, offset_m = self.cumulative_distance_at(position_local)
        return instruction_at_distance(
            self._route_index, self._trace, cumulative_m, offset_m
        )


class GeometryNavigator:
    """Navigation text classified from the route the runtime generates each step.

    This is the pre-dataset behaviour: a left turn, a right turn or straight ahead
    read off the route waypoints, with the same ``route_to_nav_text`` wording the
    drivers used to compute themselves.
    """

    def lookup(
        self, position_local: np.ndarray, route_in_rig: np.ndarray | None
    ) -> NavInstruction | None:
        """Instruction for the route ahead; None when the route has no usable waypoint."""
        if route_in_rig is None:
            return None
        text = route_to_nav_text(route_in_rig)
        if text is None:
            return None
        status = (
            NavStatus.OUTSIDE_HORIZON
            if text == "Continue straight"
            else NavStatus.UPCOMING
        )
        return NavInstruction(
            maneuver="",
            instruction_text=text,
            remaining_distance_m=0.0,
            status=status,
            route_offset_m=0.0,
        )


# Either source of turn-by-turn text a rollout can run with.
Navigator = RouteNavigator | GeometryNavigator


@dataclass(frozen=True)
class NavRouteLibrary:
    """Native route indices for the clips of one run, resolved once at startup."""

    route_indices: dict[str, Path]

    @staticmethod
    def build(route_index_dir: str, clip_ids: set[str]) -> NavRouteLibrary:
        """Discover requested clips across suite subtrees; fail on zero coverage.

        Args:
            route_index_dir: Mounted navigation version root.
            clip_ids: Source IDs selected for the run.

        Returns:
            A library containing covered clips; uncovered clips fail at rollout setup.
        """
        indices = find_route_indices(Path(route_index_dir), clip_ids)
        logger.info(
            "Navigation: %d of %d configured clips have a route index in %s",
            len(indices),
            len(clip_ids),
            route_index_dir,
        )
        if clip_ids and not indices:
            raise ValueError(
                f"none of the {len(clip_ids)} configured clips has a route index in "
                f"{route_index_dir}"
            )
        return NavRouteLibrary(route_indices=indices)

    def route_index_for_clip(self, clip_id: str) -> SessionRouteIndex:
        """Load the native session payload for a source ID.

        Args:
            clip_id: Source ID selected for the run.

        Returns:
            The clip's recorded route. Raises KeyError if the clip is uncovered.
        """
        return load_route_index(str(self.route_indices[clip_id]), clip_id)


def load_route_index_for_usdz(
    usdz_path: str | Path, route_index_dir: str
) -> SessionRouteIndex:
    """Load navigation using the archive's embedded clip ID, not its filename.

    Only metadata is read from the USDZ; scene assets are not extracted.
    Raises ValueError if the clip has no route index beneath the version root.
    """
    clip_id = Artifact(str(usdz_path)).scene_id
    library = NavRouteLibrary.build(route_index_dir, {clip_id})
    return library.route_index_for_clip(clip_id)
