# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
ASL (Alpasim Simulation Log) replay infrastructure for integration testing.

This module provides tools to record microservice interactions from ASL files
and replay them during testing to ensure refactoring preserves exact behavior.

:Warning: This module is meant only for simulations invovling a single instance
of each microservice, as its support for session management is not fully implemented.
"""

from __future__ import annotations

import difflib
import json
import logging
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Type, Union

from alpasim_grpc.v0 import common_pb2
from alpasim_grpc.v0.egodriver_pb2 import RolloutCameraImage
from alpasim_grpc.v0.sensorsim_pb2 import RGBRenderRequest
from alpasim_utils.logs import async_read_pb_log
from google.protobuf import json_format
from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message

logger = logging.getLogger(__name__)


# List of fields that are expected to be different between runs
DYNAMIC_FIELDS = {
    "sessionUuid",
    "session_uuid",
    "randomSeed",
    "random_seed",
}

# Mapping of service names to their version field names in RolloutMetadata
SERVICE_VERSION_MAP = {
    "driver": "egodriver_version",
    "physics": "physics_version",
    "trafficsim": "traffic_version",
    "controller": "controller_version",
    "sensorsim": "sensorsim_version",
    "runtime": "runtime_version",
}


def _pair_response_fifo(pending_requests: Deque[Any], _response: Any) -> Any:
    """Return the next pending request in FIFO order."""
    return pending_requests.popleft()


@dataclass
class ExchangeConfig:
    """Configuration for a request/response exchange."""

    method: str
    request_entry: str
    response: Union[
        str, Type, None
    ]  # Entry type name for paired response, or Type for direct response
    processor: Optional[Callable] = None  # Optional special processing
    response_matcher: Callable[[Deque[Any], Any], Any] = _pair_response_fifo

    @property
    def is_direct(self) -> bool:
        """Auto-detect if this is a direct exchange based on response type."""
        return not isinstance(self.response, str)


def _find_config_for_entry(
    entry_type: str,
) -> Tuple[Optional[str], Optional[ExchangeConfig]]:
    """Find the service and config for a given entry type."""
    for service, exchanges in SERVICE_EXCHANGES.items():
        for config in exchanges:
            # Check if it's a request entry
            if entry_type == config.request_entry:
                return service, config
            # Check if it's a response entry (for paired exchanges)
            if isinstance(config.response, str) and entry_type == config.response:
                return service, config
    return None, None


def _pop_deque_at_index(values: Deque[Any], index: int) -> Any:
    """Pop the element at ``index`` from a deque while preserving relative order."""
    values.rotate(-index)
    item = values.popleft()
    values.rotate(index)
    return item


def _physics_request_response_distance(request: Any, response: Any) -> float:
    """Compute squared position distance between physics request and response.

    Compares the last pose from the request ego_trajectory_aabb against
    the last pose from the response ego_trajectory_aabb.
    """
    request_vec = request.ego_data.ego_trajectory_aabb.poses[-1].pose.vec
    response_vec = response.ego_trajectory_aabb.poses[-1].pose.vec

    dx = request_vec.x - response_vec.x
    dy = request_vec.y - response_vec.y
    dz = request_vec.z - response_vec.z
    return dx * dx + dy * dy + dz * dz


def _pair_physics_response_by_closest_pose(
    pending_requests: Deque[Any], response: Any
) -> Any:
    """Pair a physics response with the closest pending request future position."""
    best_index = 0
    best_distance: Optional[float] = None

    for idx, request in enumerate(pending_requests):
        distance = _physics_request_response_distance(request, response)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_index = idx

    return _pop_deque_at_index(pending_requests, best_index)


def _render_dynamic_object_sort_key(item: dict[str, Any]) -> tuple[str, str]:
    """Build a deterministic sort key for render request dynamic objects."""
    return (item.get("trackId", ""), json.dumps(item, sort_keys=True))


# Hierarchical service configuration
SERVICE_EXCHANGES = {
    "driver": [
        ExchangeConfig(
            method="drive", request_entry="driver_request", response="driver_return"
        ),
        ExchangeConfig(
            method="start_session",
            request_entry="driver_session_request",
            response=common_pb2.SessionRequestStatus,
        ),
        ExchangeConfig(
            method="submit_image_observation",
            request_entry="driver_camera_image",
            response=common_pb2.Empty,
            processor=lambda self, entry: self.driver_images.append(entry),
        ),
        ExchangeConfig(
            method="submit_egomotion_observation",
            request_entry="driver_ego_trajectory",
            response=common_pb2.Empty,
        ),
        ExchangeConfig(
            method="submit_route",
            request_entry="route_request",
            response=common_pb2.Empty,
        ),
        ExchangeConfig(
            method="submit_recording_ground_truth",
            request_entry="ground_truth_request",
            response=common_pb2.Empty,
        ),
    ],
    "physics": [
        ExchangeConfig(
            method="ground_intersection",
            request_entry="physics_request",
            response="physics_return",
            response_matcher=_pair_physics_response_by_closest_pose,
        ),
    ],
    "trafficsim": [
        ExchangeConfig(
            method="simulate",
            request_entry="traffic_request",
            response="traffic_return",
        ),
        ExchangeConfig(
            method="start_session",
            request_entry="traffic_session_request",
            response=common_pb2.SessionRequestStatus,
        ),
    ],
    "controller": [
        ExchangeConfig(
            method="run_controller_and_vehicle",
            request_entry="controller_request",
            response="controller_return",
        ),
    ],
    "sensorsim": [
        ExchangeConfig(
            method="render_rgb",
            request_entry="render_request",
            response=None,  # Special case - no response in ASL
        ),
        ExchangeConfig(
            method="get_available_cameras",
            request_entry="available_cameras_request",
            response="available_cameras_return",
        ),
        ExchangeConfig(
            method="get_available_ego_masks",
            request_entry="available_ego_masks_request",
            response="available_ego_masks_return",
        ),
    ],
}


class ASLReader:
    """Reader for ASL files with request/response pairing"""

    def __init__(
        self,
        asl_file_path: str,
    ) -> None:
        self.asl_file_path = asl_file_path
        # Store request/response pairs for each service.method combination
        self._exchanges: dict[str, list[tuple[Message, Optional[Message]]]] = {}
        # Track which exchanges have been consumed during replay
        self._consumed_indices: dict[str, set[int]] = {}
        # Store ASL metadata like configuration and rollout info
        self.asl_metadata: Dict[str, Any] = {}
        # Store camera images from driver for sensorsim correlation
        self.driver_images: List[RolloutCameraImage] = []

    def reset(self) -> None:
        """Reset the ASL reader for a fresh load"""
        self._exchanges.clear()
        self._consumed_indices.clear()
        self.asl_metadata.clear()
        self.driver_images.clear()

    async def load_exchanges(self) -> None:
        """Load and pair request/response exchanges from ASL file"""
        pending_requests: Dict[str, Deque[Any]] = defaultdict(deque)
        self.reset()

        logger.info(f"Loading exchanges from ASL file: {self.asl_file_path}")

        async for log_entry in async_read_pb_log(self.asl_file_path):
            entry_type = log_entry.WhichOneof("log_entry")

            # Handle special non-exchange entries
            if entry_type == "rollout_metadata":
                self.asl_metadata["rollout_metadata"] = log_entry.rollout_metadata
                continue
            elif entry_type in ["actor_poses", "egomotion_estimate_error"]:
                # Currently not used in replay
                continue

            # Find configuration for this entry type
            service, config = _find_config_for_entry(entry_type)
            if not config:
                raise AssertionError(f"Unknown entry type: {entry_type}")
            assert service is not None  # Guaranteed by _find_config_for_entry

            entry_data = getattr(log_entry, entry_type)

            # Handle based on whether it's a request or response
            if entry_type == config.request_entry:
                # This is a request
                if config.is_direct:
                    # Direct exchange - create immediately
                    response_factory = config.response
                    assert not isinstance(response_factory, str)
                    response = response_factory() if response_factory else None
                    self._add_exchange(service, config.method, entry_data, response)
                else:
                    # Paired exchange - queue the request
                    queue_key = f"{service}.{config.method}"
                    pending_requests[queue_key].append(entry_data)
            else:
                # This is a response - pop matching request
                queue_key = f"{service}.{config.method}"
                pending_queue = pending_requests[queue_key]
                if not pending_queue:
                    raise AssertionError(
                        f"Response without pending request for {queue_key}"
                    )

                request = config.response_matcher(pending_queue, entry_data)
                self._add_exchange(service, config.method, request, entry_data)

            # Run any special processing
            if config.processor:
                config.processor(self, entry_data)

        # Check for unmatched requests
        unmatched_requests = {
            key: len(queue) for key, queue in pending_requests.items() if queue
        }
        if unmatched_requests:
            for key, count in unmatched_requests.items():
                logger.warning(f"  {key}: {count} unmatched requests")
            raise AssertionError(f"Unmatched requests: {unmatched_requests}")

        # Log summary
        total_exchanges = sum(len(exchanges) for exchanges in self._exchanges.values())
        service_details = "\n  " + "\n  ".join(
            f"{service}: {len(exchanges)} exchanges"
            for service, exchanges in self._exchanges.items()
        )
        logger.info(
            "Loaded %d exchanges across %d services: %s",
            total_exchanges,
            len(self._exchanges),
            service_details,
        )
        logger.info(
            "Additional data: %d images",
            len(self.driver_images),
        )

    def _add_exchange(
        self, service: str, method: str, request: Any, response: Any
    ) -> None:
        """Add a request/response pair to the replay data"""
        key = f"{service}.{method}"
        if key not in self._exchanges:
            self._exchanges[key] = []
            self._consumed_indices[key] = set()

        self._exchanges[key].append((request, response))

    def get_exchanges(
        self, service: str, method: str
    ) -> list[tuple[Message, Optional[Message]]]:
        """Return recorded exchanges for a service.method pair."""
        return self._exchanges.get(f"{service}.{method}", [])

    def get_exchange_summary(self) -> Dict[str, Any]:
        """Get summary of exchanges loaded and consumed"""
        summary: Dict[str, Any] = {}
        for key, exchanges in self._exchanges.items():
            consumed_set = self._consumed_indices.get(key, set())
            consumed_count = len(consumed_set)
            total = len(exchanges)

            result: Dict[str, Any] = {
                "total": total,
                "consumed": consumed_count,
                "remaining": total - consumed_count,
            }

            # Add unconsumed indices if any messages remain
            if consumed_count < total:
                all_indices = set(range(total))
                unconsumed_indices = sorted(all_indices - consumed_set)

                # Format unconsumed indices with ellipsis for long lists
                if len(unconsumed_indices) > 10:
                    displayed_indices = unconsumed_indices[:10]
                    result["unconsumed_indices"] = str(displayed_indices) + " ['...']"
                else:
                    result["unconsumed_indices"] = unconsumed_indices

                result["unconsumed_count"] = total - consumed_count

            summary[key] = result
        return summary

    def find_and_consume_matching_request(
        self, request: Any, service: str, method: str
    ) -> Optional[Tuple[int, Any]]:
        """Find a matching request and mark it as consumed.

        Args:
            request: The incoming request to match
            service: The service name
            method: The method name

        Returns:
            Tuple of (index, recorded_response) if found, None otherwise

        Raises:
            KeyError: If no exchanges are recorded for this service.method
        """
        key = f"{service}.{method}"
        exchanges = self._exchanges[key]

        # Find matching request
        match_result = self._find_matching_request(request, key, exchanges)

        if match_result:
            match_index, unused_recorded_response = match_result

            self._consumed_indices[key].add(match_index)

            return match_result

        return None

    def _find_matching_request(
        self, request: Any, key: str, exchanges: list, max_lookahead: int = 50
    ) -> Optional[Tuple[int, Any]]:
        """Find a matching request in the exchanges list, skipping consumed ones."""
        consumed = self._consumed_indices.get(key, set())

        # Get unconsumed indices within the lookahead window
        start_index = next(
            (i for i in range(len(exchanges)) if i not in consumed), None
        )
        if start_index is None:
            return None

        # Only check unconsumed messages within lookahead window
        for idx in range(start_index, min(start_index + max_lookahead, len(exchanges))):
            if idx in consumed:
                continue

            expected_request, recorded_response = exchanges[idx]
            if self.requests_match(request, expected_request):
                return (idx, recorded_response)

        return None

    def generate_no_match_error(
        self,
        request: Any,
        service: str,
        method: str,
    ) -> str:
        """Generate detailed error message when no matching request is found."""
        key = f"{service}.{method}"
        exchanges = self._exchanges.get(key, [])
        consumed = self._consumed_indices.get(key, set())
        unconsumed = [i for i in range(len(exchanges)) if i not in consumed]

        # Generate detailed error with unconsumed message info
        if unconsumed:
            # Show diff with first unconsumed message
            idx = unconsumed[0]
            expected_request, _ = exchanges[idx]
            diff = self.generate_diff(expected_request, request)[:1000]

            error_msg = (
                f"Request not found in {key}\n"
                f"Consumed indices: {sorted(consumed)}\n"
                f"Unconsumed indices: {unconsumed[:10]}...\n"  # Show first 10
                f"Diff with first unconsumed (index {idx}):\n{diff}"
            )
        else:
            error_msg = f"All {len(exchanges)} exchanges already consumed for {key}"

        return error_msg

    def get_driver_image_for_camera(
        self,
        camera_id: str,
        timestamp_us: int = 0,
    ) -> Optional[bytes]:
        """Get driver camera image data that corresponds to a render request.

        This is needed because we read in the camera images from the logged
        driver messages, not from the rendering returns (which aren't logged).

        Args:
            camera_id: The camera ID to match (logical name like "camera_front_wide_120fov")
            timestamp_us: The timestamp to match (frame_start_us from render request)

        Returns:
            The image bytes if found, None otherwise
        """

        camera_images = [
            image
            for image in self.driver_images
            if image.camera_image.logical_id == camera_id
        ]

        if not camera_images:
            logger.error("No image found for camera %s", camera_id)
            return None

        # If timestamp is 0, return the first matching camera.
        if timestamp_us == 0:
            return camera_images[0].camera_image.image_bytes

        # Find exact/closest timestamp match.
        best_match = min(
            camera_images,
            key=lambda image: abs(image.camera_image.frame_start_us - timestamp_us),
        )
        best_time_diff = abs(best_match.camera_image.frame_start_us - timestamp_us)

        if best_time_diff == 0:
            return best_match.camera_image.image_bytes

        earliest_timestamp_us = min(
            image.camera_image.frame_start_us for image in camera_images
        )

        # Warmup render can occur before the first logged driver frame.
        if timestamp_us < earliest_timestamp_us:
            logger.info(
                "Expected warmup timestamp miss for camera %s at %d us "
                "(first available: %d us). Returning first available frame.",
                camera_id,
                timestamp_us,
                earliest_timestamp_us,
            )
            return best_match.camera_image.image_bytes

        logger.error(
            "Unexpected non-exact timestamp for camera %s at %d us "
            "(closest available: %d us, diff: %d us).",
            camera_id,
            timestamp_us,
            best_match.camera_image.frame_start_us,
            best_time_diff,
        )
        return None

    def generate_diff(self, expected: Any, actual: Any) -> str:
        """Generate a detailed diff between expected and actual messages"""

        # Convert protobuf messages to dict for comparison

        if isinstance(expected, Message):
            expected_dict = self._normalize_request_dict(expected)
        else:
            expected_dict = MessageToDict(expected) if expected else {}

        if isinstance(actual, Message):
            actual_dict = self._normalize_request_dict(actual)
        else:
            actual_dict = MessageToDict(actual) if actual else {}

        expected_json = json.dumps(expected_dict, indent=2, sort_keys=True).splitlines()
        actual_json = json.dumps(actual_dict, indent=2, sort_keys=True).splitlines()

        diff = difflib.unified_diff(
            expected_json,
            actual_json,
            fromfile="expected",
            tofile="actual",
            lineterm="",
        )

        return "\n".join(diff)

    def requests_match(self, actual: Any, expected: Any) -> bool:
        """Compare requests, ignoring dynamic fields and tolerating float drift.

        Floating-point values may differ by up to ~1 ULP after proto→Pose→proto
        round-trips (e.g. quaternion normalization).  We use approximate
        comparison for floats to avoid false negatives.
        """
        # If they're exactly equal, no need for complex comparison
        if actual == expected:
            return True

        # For protobuf messages, do field-by-field comparison
        if isinstance(actual, Message) and isinstance(expected, Message):
            # Convert to dicts for easier manipulation
            actual_dict = self._normalize_request_dict(actual)
            expected_dict = self._normalize_request_dict(expected)

            actual_normalized = _remove_dynamic_fields(actual_dict)
            expected_normalized = _remove_dynamic_fields(expected_dict)

            return _approx_equal(actual_normalized, expected_normalized)

        # For non-protobuf messages, use direct comparison
        return actual == expected

    def _normalize_request_dict(self, request: Message) -> dict[str, Any]:
        """Normalize request messages for comparison."""
        request_dict = json_format.MessageToDict(request)

        if isinstance(request, RGBRenderRequest):
            dynamic_objects = request_dict.get("dynamicObjects")
            if isinstance(dynamic_objects, list):
                request_dict["dynamicObjects"] = sorted(
                    dynamic_objects,
                    key=_render_dynamic_object_sort_key,
                )

        return request_dict

    def get_map_id(self) -> str:
        """Get the scene ID from the ASL metadata"""
        scene_id = self.asl_metadata["rollout_metadata"].session_metadata.scene_id
        pattern = r"^clipgt-([0-9a-fA-F-]{36})$"
        match = re.match(pattern, scene_id)
        if match is None:
            raise ValueError(
                f"Scene ID '{scene_id}' does not match expected pattern 'clipgt-<uuid>'"
            )
        return match.group(1)  # Map id (without clipgt prefix)

    def is_complete(self) -> bool:
        """Check if all messages consumed (ignoring skipped)"""
        for key, exchanges in self._exchanges.items():
            # Check if all exchanges for this key have been consumed
            consumed_set = self._consumed_indices.get(key, set())
            if len(consumed_set) < len(exchanges):
                return False
        return True

    def get_service_version(self, service_name: str) -> Optional[Message]:
        """Get the version information for a specific service from ASL metadata.

        Args:
            service_name: The service name (e.g., "driver", "physics", "trafficsim")

        Returns:
            The VersionId message for the service, or None if not found
        """
        return getattr(
            self.asl_metadata["rollout_metadata"].version_ids,
            SERVICE_VERSION_MAP[service_name],
        )


# Remove dynamic fields from both dicts
def _remove_dynamic_fields(d: Any) -> Any:
    if isinstance(d, dict):
        return {
            k: _remove_dynamic_fields(v)
            for k, v in d.items()
            if k not in DYNAMIC_FIELDS
        }
    elif isinstance(d, list):
        return [_remove_dynamic_fields(item) for item in d]
    return d


# Tolerances for float comparison.  Simulation state accumulates small
# float-precision differences across steps (e.g. quaternion normalization,
# spline resampling), so route waypoints and poses drift over the 60-step
# replay.  rel_tol=1e-3 and abs_tol=1e-4 provide ~100× headroom over the
# largest observed per-step drift while still catching gross regressions.
_FLOAT_REL_TOL = 1e-3
_FLOAT_ABS_TOL = 1e-4


def _approx_equal(a: Any, b: Any) -> bool:
    """Recursively compare two structures with approximate float matching.

    Handles protobuf ``MessageToDict`` quirks: default-valued fields (0, 0.0,
    "", False) are omitted from the dict, so a near-zero float in one dict
    may correspond to a missing key in the other.
    """
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        all_keys = a.keys() | b.keys()
        for k in all_keys:
            va = a.get(k)
            vb = b.get(k)
            if va is None and isinstance(vb, float):
                # Key absent in a → protobuf default 0.0
                if not math.isclose(0.0, vb, abs_tol=_FLOAT_ABS_TOL):
                    return False
            elif vb is None and isinstance(va, float):
                if not math.isclose(va, 0.0, abs_tol=_FLOAT_ABS_TOL):
                    return False
            elif va is None or vb is None:
                # Non-float key present in only one dict
                return False
            elif not _approx_equal(va, vb):
                return False
        return True
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(_approx_equal(ai, bi) for ai, bi in zip(a, b))
    if isinstance(a, float):
        return math.isclose(a, b, rel_tol=_FLOAT_REL_TOL, abs_tol=_FLOAT_ABS_TOL)
    return a == b
