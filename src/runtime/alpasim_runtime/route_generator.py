# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import logging
import math
from abc import ABC, abstractmethod
from typing import final

import numpy as np
from alpasim_runtime.config import RouteGeneratorType
from alpasim_utils.geometry import Polyline, Pose
from trajdata.maps import VectorMap
from trajdata.maps.vec_map_elements import RoadLane

logger = logging.getLogger(__name__)


class RouteGenerator(ABC):
    """
    The RouteGenerator is a base class with common functionality for generating routes. It is
    the responsibility of the subclasses to populate the local-frame waypoints and optionally
    to extend the waypoints when the end of the recorded trajectory is reached. The RouteGenerator
    is responsible for generating the waypoints for the provided pose, populating waypoints up to
    80m from the current position.
    """

    NUM_WAYPOINTS: int = 20
    DISTANCE_BETWEEN_WAYPOINTS: float = 80.0 / (
        NUM_WAYPOINTS - 1.0
    )  # 80 m desired length of traj
    BAD_LANE_CONNECTION_ANGLE_THRESHOLD_DEG: float = (
        150.0  # [deg] maximum angle for lane connections
    )

    @classmethod
    def create(
        cls,
        recorded_waypoints_in_local: np.ndarray,
        vector_map: VectorMap,
        route_generator_type: RouteGeneratorType,
    ) -> "RouteGenerator":
        """
        Factory method to create a RouteGenerator
        Args:
          recorded_waypoints_in_local: the waypoints in the local frame. (N, 3) array
          vector_map: the map data
          route_generator_type: the type of route generator to create
        Returns:
          A route generator of the specified type
        """
        if route_generator_type == RouteGeneratorType.RECORDED:
            return RouteGeneratorRecorded(recorded_waypoints_in_local)
        elif route_generator_type == RouteGeneratorType.MAP:
            return RouteGeneratorMap(recorded_waypoints_in_local, vector_map)
        else:
            raise ValueError(f"Invalid route generator type: {route_generator_type}")

    def __init__(self, rig_waypoints_in_local: np.ndarray) -> None:
        """
        Initialize the route generator with the rig waypoints in the local frame
        Args:
          rig_waypoints_in_local: the waypoints in the local frame. (N, 3) array
        """
        if len(rig_waypoints_in_local) < 2:
            raise ValueError("At least two waypoints are required")
        self._route_polyline_in_local = Polyline(points=rig_waypoints_in_local.copy())
        # Note: downsample to remove redundant/noisy waypoints, especially when the vehicle is
        # approximately stationary during recording
        DOWNSAMPLE_MIN_DISTANCE_M: float = 0.2  # [m]
        self._route_polyline_in_local.downsample_with_min_distance(
            DOWNSAMPLE_MIN_DISTANCE_M
        )
        RouteGenerator.sanity_check_waypoints_for_foldback(
            self._route_polyline_in_local.points
        )

    @property
    def route_polyline_in_local(self) -> Polyline:
        """Route geometry expressed in the local frame."""
        return self._route_polyline_in_local

    @route_polyline_in_local.setter
    def route_polyline_in_local(
        self, route_polyline_in_local: np.ndarray | Polyline
    ) -> None:
        """Set the route geometry expressed in the local frame."""
        if isinstance(route_polyline_in_local, Polyline):
            self._route_polyline_in_local = route_polyline_in_local
        else:
            self._route_polyline_in_local = Polyline(points=route_polyline_in_local)

    def generate_route(self, timestamp_us: int, pose_local_to_rig: Pose) -> Polyline:
        """
        Generate the route polyline in the rig frame for the given timestamp and pose
        Args:
          timestamp_us: timestamp in microseconds
          pose_local_to_rig: pose of the local frame to the rig frame
        Returns:
          a Polyline with waypoints in the rig frame
        """
        # First ensure we have enough waypoints in the local polyline
        self._ensure_sufficient_waypoints(pose_local_to_rig.vec3)

        # Resample the stored route starting from the current pose
        resampled_route_in_local = self._route_polyline_in_local.resample_from_point(
            start_point=pose_local_to_rig.vec3,
            spacing=self.DISTANCE_BETWEEN_WAYPOINTS,
            n_points=self.NUM_WAYPOINTS,
        )

        # Transform route geometry to the rig frame
        route_polyline_in_rig = resampled_route_in_local.transform(
            pose_local_to_rig.inverse()
        )

        route_polyline_in_rig = route_polyline_in_rig.zero_out_z()

        return route_polyline_in_rig

    def _ensure_sufficient_waypoints(self, current_position: np.ndarray) -> None:
        """
        Ensure the stored route polyline has sufficient waypoints from the current position.
        Extends the route if necessary to have at least 80m of waypoints.
        Args:
          current_position: Current position in local frame (3,) array
        """
        # Get cumulative distances from the projected point
        cumulative_from_point, distance_to_projection = (
            self._route_polyline_in_local.get_cumulative_distances_from_point(
                current_position
            )
        )

        if len(cumulative_from_point) == 0:
            return

        # Calculate remaining distance from projected point to end of the route
        remaining_distance = cumulative_from_point[-1]

        # Required distance for our waypoints
        required_distance = self.DISTANCE_BETWEEN_WAYPOINTS * (self.NUM_WAYPOINTS - 1)

        # Keep extending until we have enough distance
        while remaining_distance < required_distance:
            if not self._extend_waypoints():
                break

            # Sanity check the extended waypoints
            RouteGenerator.sanity_check_waypoints_for_foldback(
                self._route_polyline_in_local.points
            )

            # Recalculate remaining distance after extension
            cumulative_from_point, _ = (
                self._route_polyline_in_local.get_cumulative_distances_from_point(
                    current_position
                )
            )
            remaining_distance = cumulative_from_point[-1]

    @abstractmethod
    def _extend_waypoints(self) -> bool:
        """Extend the waypoints. Returns True if successful, False if cannot extend further."""
        pass

    @staticmethod
    def prepare_for_policy(polyline: Polyline) -> Polyline:
        """
        The target policy expects a few conditions to be true for the route waypoints:
        - There are always NUM_WAYPOINTS waypoints
        - If there are not enough waypoints, the remaining waypoints are NaNs
        - The distances between waypoints are consistent [3.5, 4.5] m apart
        This method ensures these three conditions are met and pads the polyline with NaNs
        if necessary.
        Args:
          polyline: the polyline to process
        Returns:
          a new Polyline with the required conditions met
        """
        MIN_DIST_M = 3.5
        MAX_DIST_M = 4.5

        points = polyline.waypoints.copy()
        segment_lengths = polyline.segment_lengths
        invalid_indices = (segment_lengths < MIN_DIST_M) | (
            segment_lengths > MAX_DIST_M
        )

        if np.any(invalid_indices):
            first_invalid_index = np.where(invalid_indices)[0][0]
            points = points[0 : (first_invalid_index + 1)]

        # pad the polyline with NaNs if necessary
        num_waypoints_to_pad = RouteGenerator.NUM_WAYPOINTS - len(points)
        if num_waypoints_to_pad > 0:
            nan_padding = np.full((num_waypoints_to_pad, 3), np.nan)
            points = np.vstack((points, nan_padding))
        elif num_waypoints_to_pad < 0:
            points = points[0 : RouteGenerator.NUM_WAYPOINTS]

        return Polyline(points=points)

    @staticmethod
    def sanity_check_waypoints_for_foldback(waypoints: np.ndarray) -> None:
        """
        Check that the waypoints do not fold back on themselves, which can
        happen if the map is somehow flawed.
        Args:
          waypoints: the waypoints to check (N, 3) array
        Raises:
          ValueError: if the waypoints have an abrupt/inconsistent direction change
        """
        for i in range(1, len(waypoints) - 1):
            v1 = waypoints[i] - waypoints[i - 1]
            v2 = waypoints[i + 1] - waypoints[i]
            if np.linalg.norm(v1) < 1.0e-3 or np.linalg.norm(v2) < 1.0e-3:
                continue
            v1_normalized = v1 / np.linalg.norm(v1)
            v2_normalized = v2 / np.linalg.norm(v2)
            dot_product = np.dot(v1_normalized, v2_normalized)
            angle_deg = math.degrees(math.acos(np.clip(dot_product, -1.0, 1.0)))
            if angle_deg > RouteGenerator.BAD_LANE_CONNECTION_ANGLE_THRESHOLD_DEG:
                raise ValueError(
                    f"Waypoint {i} fails sanity check: route folds back on itself with angle {angle_deg:.2f} deg"
                )


class RouteGeneratorRecorded(RouteGenerator):
    """
    A version of the RouteGenerator that uses the recorded rig waypoints to
    determine the route.
    """

    def __init__(self, recorded_waypoints_in_local: np.ndarray) -> None:
        """
        Initialize the route generator (recorded waypoints are passed through unmodified)
        Args:
          recorded_waypoints_in_local: the waypoints in the local frame. (N, 3) array
        """
        super().__init__(recorded_waypoints_in_local)

    def _extend_waypoints(self) -> bool:
        """Recorded routes cannot be extended beyond the recording."""
        return False


@final
class RouteGeneratorMap(RouteGenerator):
    """
    A version of the RouteGenerator that uses the map data and original recorded
    trajectory to generate the route. Once the recorded trajectory runs out, the
    route is extended by following the center of the lane at the end of the recorded
    trajectory.
    """

    OFF_MAP_THRESHOLD_M: float = 10.0  # [m] maximum distance to the original trajectory

    def __init__(
        self, recorded_waypoints_in_local: np.ndarray, vector_map: VectorMap
    ) -> None:
        """
        Initialize the route generator with the rig waypoints in the local frame
        and the map data.
        Args:
          recorded_waypoints_in_local: the waypoints in the local frame. (N, 3) array
          vector_map: the map data
        """
        self._map = vector_map
        self._current_lane = None
        super().__init__(self._determine_waypoints(recorded_waypoints_in_local))

    def _determine_waypoints(
        self, recorded_waypoints_in_local: np.ndarray
    ) -> np.ndarray:
        # For each waypoint in the recorded rig trajectory, find the possible lanes
        possible_lane_ids: list[list[str]] = []
        heading = np.array([0.0])
        for i in range(len(recorded_waypoints_in_local)):
            if i != len(recorded_waypoints_in_local) - 1:
                relative_motion = (
                    recorded_waypoints_in_local[i + 1] - recorded_waypoints_in_local[i]
                )[:2]
                if np.linalg.norm(relative_motion) > 1.0e-2:
                    heading = np.array(
                        [math.atan2(relative_motion[1], relative_motion[0])]
                    )

            possible_lane_ids.append(
                [
                    road_lane.id
                    for road_lane in self._map.get_current_lane(
                        np.concatenate((recorded_waypoints_in_local[i], heading)),
                        max_dist=4.0,
                    )
                ]
            )
            if (
                len(possible_lane_ids[-1]) == 0
            ):  # if no lane is found, add the closest lane
                possible_lane_ids[-1].append(
                    self._map.get_closest_lane(recorded_waypoints_in_local[i]).id
                )

        best_lane_ids = self._determine_best_lane_sequence(possible_lane_ids)

        # For each waypoint in the recorded rig trajectory, project to the lane center
        resolved_waypoints_in_local = []
        _current_lane = self._map.get_road_lane(best_lane_ids[0])
        for i in range(len(recorded_waypoints_in_local)):
            waypoint_in_local = recorded_waypoints_in_local[i]

            if _current_lane.id != best_lane_ids[i]:
                _current_lane = self._map.get_road_lane(best_lane_ids[i])

            projected_point_in_local = _current_lane.center.project_onto(
                waypoint_in_local.reshape(1, 3)
            )[0, 0:3]
            resolved_waypoints_in_local.append(projected_point_in_local)
            offset = np.linalg.norm(waypoint_in_local - projected_point_in_local)
            if offset > self.OFF_MAP_THRESHOLD_M:
                raise ValueError(
                    f"Waypoint {i} does not pass sanity check: route too far from original trajectory "
                    f"{offset=} > {self.OFF_MAP_THRESHOLD_M=}"
                )

        rig_waypoints_in_local = np.array(resolved_waypoints_in_local)

        self._current_lane = _current_lane

        return rig_waypoints_in_local

    def _extend_waypoints(self) -> bool:
        """Extend waypoints using map data. Returns True if successful, False if cannot extend."""
        # Add waypoints through the end of the current lane and reset the
        # current lane to the next lane
        if self._current_lane is None:
            return False

        _, starting_index = self._current_lane.center.project_onto(
            self._route_polyline_in_local.waypoints[-1].reshape(1, 3),
            return_index=True,
        )

        # Add waypoints from the current lane
        extension_waypoints = self._current_lane.center.points[
            (starting_index[0] + 1) :, 0:3  # noqa E203
        ]

        extended = False
        if len(extension_waypoints) > 0:
            # Create a polyline with the extension waypoints and append
            extension_polyline = Polyline(points=extension_waypoints)
            self._route_polyline_in_local = self._route_polyline_in_local.append(
                extension_polyline
            )
            extended = True

        if len(self._current_lane.next_lanes) == 0:
            self._current_lane = None
        elif len(self._current_lane.next_lanes) == 1 or (
            len(self._current_lane.center.points[-1]) < 4
        ):
            # only one lane or no heading info
            self._current_lane = self._map.get_road_lane(
                min(self._current_lane.next_lanes)
            )
        else:
            # multiple next lanes, choose the best one
            # arbitrarily pick the next lane with the closest heading
            # in the future, we can make this more sophisticated
            current_lane_heading = self._current_lane.center.points[-1][3]
            min_heading_diff = float("inf")
            for next_lane_id in self._current_lane.next_lanes:
                next_lane = self._map.get_road_lane(next_lane_id)
                next_lane_heading = next_lane.center.points[0][3]
                heading_diff = abs(current_lane_heading - next_lane_heading)
                heading_diff = min(
                    heading_diff, 2 * math.pi - heading_diff
                )  # wrap around
                if heading_diff < min_heading_diff:
                    min_heading_diff = heading_diff
                    self._current_lane = next_lane

        return extended

    def _determine_best_lane_sequence(
        self, possible_lane_ids: list[list[str]]
    ) -> list[str]:
        # aim for consistency in lane choice
        best_lane_sequence = []

        best_lane_sequence.append(
            RouteGeneratorMap.find_consistent_upcoming_lane_id(possible_lane_ids)
        )

        for i in range(1, len(possible_lane_ids)):
            if best_lane_sequence[-1] in possible_lane_ids[i]:
                best_lane_sequence.append(best_lane_sequence[-1])
            else:
                # Previous lane is no longer valid, first see if there is an obvious
                # continuation through looking at the intersection of the previous lane's
                # continuations with the current lane possibilities
                intersection = list(
                    set(possible_lane_ids[i])
                    & set(self._map.get_road_lane(best_lane_sequence[-1]).next_lanes)
                )
                if len(intersection) == 1:  # the choice is obvious
                    best_lane_sequence.append(intersection[0])
                else:
                    best_lane_sequence.append(
                        RouteGeneratorMap.find_consistent_upcoming_lane_id(
                            possible_lane_ids[i:]
                        )
                    )

        return best_lane_sequence

    @staticmethod
    def find_consistent_upcoming_lane_id(
        upcoming_lane_ids: list[list[str]], current_lane: RoadLane = None
    ) -> str:
        # Prefer the lane id that is common to all upcoming waypoints
        potential_lane_ids = upcoming_lane_ids[0].copy()
        if len(potential_lane_ids) == 1:
            return potential_lane_ids[0]
        for i in range(1, len(upcoming_lane_ids)):
            for potential_lane_id in potential_lane_ids.copy():
                if potential_lane_id not in upcoming_lane_ids[i]:
                    potential_lane_ids.remove(potential_lane_id)
                    if len(potential_lane_ids) <= 1:
                        return potential_lane_ids[0]
        logger.warning("Unable to find a preferable lane id, returning the first one")
        return potential_lane_ids[0]
