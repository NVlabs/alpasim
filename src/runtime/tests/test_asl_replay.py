# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Unit tests for ASL replay infrastructure."""

from collections.abc import AsyncIterator
from typing import Any, Dict, cast
from unittest.mock import Mock

import pytest
from alpasim_grpc.v0.common_pb2 import Empty, Pose, PoseAtTime, Trajectory, Vec3
from alpasim_grpc.v0.egodriver_pb2 import RolloutCameraImage, RouteRequest
from alpasim_grpc.v0.logging_pb2 import LogEntry
from alpasim_grpc.v0.physics_pb2 import (
    PhysicsGroundIntersectionRequest,
    PhysicsGroundIntersectionReturn,
)
from alpasim_grpc.v0.sensorsim_pb2 import (
    AvailableEgoMasksReturn,
    DynamicObject,
    EgoMaskId,
    PosePair,
    RGBRenderRequest,
)
from alpasim_grpc.v0.traffic_pb2 import TrafficRequest, TrafficReturn
from alpasim_runtime.replay_services.asl_reader import ASLReader, _remove_dynamic_fields


def _build_physics_request(now_us: int, future_us: int, future_x: float) -> Any:
    request = PhysicsGroundIntersectionRequest(now_us=now_us, future_us=future_us)
    request.ego_data.ego_trajectory_aabb.CopyFrom(
        Trajectory(
            poses=[
                PoseAtTime(
                    pose=Pose(vec=Vec3(x=future_x)),
                    timestamp_us=future_us,
                )
            ]
        )
    )
    return request


def _build_physics_return(ego_x: float) -> Any:
    response = PhysicsGroundIntersectionReturn()
    response.ego_trajectory_aabb.CopyFrom(
        Trajectory(poses=[PoseAtTime(pose=Pose(vec=Vec3(x=ego_x)))])
    )
    return response


class TestLoadExchangesPairing:
    @staticmethod
    async def _fake_async_read(log_entries: list[LogEntry]) -> AsyncIterator[LogEntry]:
        for entry in log_entries:
            yield entry

    def _patch_asl_reader(
        self, monkeypatch: pytest.MonkeyPatch, entries: list[LogEntry]
    ) -> None:
        """Patch async_read_pb_log to yield the given entries."""

        async def fake_async_read_pb_log(_path: str) -> AsyncIterator[LogEntry]:
            async for entry in self._fake_async_read(entries):
                yield entry

        monkeypatch.setattr(
            "alpasim_runtime.replay_services.asl_reader.async_read_pb_log",
            fake_async_read_pb_log,
        )

    @pytest.mark.asyncio
    async def test_pairs_physics_responses_by_pose_when_out_of_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        request_a = _build_physics_request(10, 20, future_x=1.0)
        request_b = _build_physics_request(20, 30, future_x=10.0)
        response_a = _build_physics_return(ego_x=1.001)
        response_b = _build_physics_return(ego_x=10.001)

        entries = [
            LogEntry(physics_request=request_a),
            LogEntry(physics_request=request_b),
            LogEntry(physics_return=response_b),
            LogEntry(physics_return=response_a),
        ]

        self._patch_asl_reader(monkeypatch, entries)

        reader = ASLReader("dummy.asl")
        await reader.load_exchanges()

        exchanges = reader._exchanges["physics.ground_intersection"]
        assert len(exchanges) == 2
        for request, response in exchanges:
            request = cast(PhysicsGroundIntersectionRequest, request)
            response = cast(PhysicsGroundIntersectionReturn, response)
            request_x = request.ego_data.ego_trajectory_aabb.poses[-1].pose.vec.x
            response_x = response.ego_trajectory_aabb.poses[-1].pose.vec.x
            assert request_x == pytest.approx(response_x, abs=0.01)

    @pytest.mark.asyncio
    async def test_non_physics_pairing_remains_fifo(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        request_a = TrafficRequest(session_uuid="a", time_query_us=100)
        request_b = TrafficRequest(session_uuid="b", time_query_us=200)
        response_1 = TrafficReturn()
        response_2 = TrafficReturn()

        entries = [
            LogEntry(traffic_request=request_a),
            LogEntry(traffic_request=request_b),
            LogEntry(traffic_return=response_1),
            LogEntry(traffic_return=response_2),
        ]

        self._patch_asl_reader(monkeypatch, entries)

        reader = ASLReader("dummy.asl")
        await reader.load_exchanges()

        exchanges = reader._exchanges["trafficsim.simulate"]
        assert [
            cast(TrafficRequest, request).time_query_us for request, _ in exchanges
        ] == [100, 200]

    @pytest.mark.asyncio
    async def test_ego_masks_exchange_is_paired(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that get_available_ego_masks request/response are paired correctly."""
        request = Empty()

        response = AvailableEgoMasksReturn()
        mask_meta = response.ego_mask_metadata.add()
        mask_meta.ego_mask_id.CopyFrom(
            EgoMaskId(
                camera_logical_id="camera_front_wide_120fov",
                rig_config_id="hyperion_8.0",
            )
        )

        entries = [
            LogEntry(available_ego_masks_request=request),
            LogEntry(available_ego_masks_return=response),
        ]

        self._patch_asl_reader(monkeypatch, entries)

        reader = ASLReader("dummy.asl")
        await reader.load_exchanges()

        exchanges = reader.get_exchanges("sensorsim", "get_available_ego_masks")
        assert len(exchanges) == 1
        recorded_request, recorded_response = exchanges[0]
        assert recorded_request == request
        recorded_response = cast(AvailableEgoMasksReturn, recorded_response)
        assert len(recorded_response.ego_mask_metadata) == 1
        assert (
            recorded_response.ego_mask_metadata[0].ego_mask_id.camera_logical_id
            == "camera_front_wide_120fov"
        )

    @pytest.mark.asyncio
    async def test_response_without_pending_request_raises_clear_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entries = [LogEntry(physics_return=_build_physics_return(ego_x=1.0))]

        self._patch_asl_reader(monkeypatch, entries)

        reader = ASLReader("dummy.asl")
        with pytest.raises(
            AssertionError,
            match="Response without pending request for physics.ground_intersection",
        ):
            await reader.load_exchanges()


class TestRequestMatching:
    """Test request matching with dynamic field handling."""

    def test_requests_match_ignores_dynamic_fields(self) -> None:
        """Test that dynamic fields are ignored during matching."""
        reader = ASLReader("dummy.asl")

        # Create messages with different session UUIDs
        msg1 = RouteRequest()
        msg1.session_uuid = "uuid-123"
        msg1.route.timestamp_us = 1000000

        msg2 = RouteRequest()
        msg2.session_uuid = "uuid-456"  # Different - should be ignored
        msg2.route.timestamp_us = 1000000  # Same

        assert reader.requests_match(msg1, msg2)

    def test_requests_match_nested_dynamic_fields(self) -> None:
        """Test dynamic field removal in nested structures."""
        # Create complex nested messages
        dict1: Dict[str, Any] = {
            "session_uuid": "uuid-1",
            "data": {
                "randomSeed": 123,
                "values": [1, 2, 3],
                "nested": {"sessionUuid": "nested-uuid-1", "important_field": "same"},
            },
        }

        dict2: Dict[str, Any] = {
            "session_uuid": "uuid-2",  # Different
            "data": {
                "randomSeed": 456,  # Different
                "values": [1, 2, 3],
                "nested": {
                    "sessionUuid": "nested-uuid-2",  # Different
                    "important_field": "same",
                },
            },
        }

        # Test the helper function directly
        cleaned1 = _remove_dynamic_fields(dict1)
        cleaned2 = _remove_dynamic_fields(dict2)

        assert cleaned1 == cleaned2
        assert "session_uuid" not in cleaned1
        assert "randomSeed" not in cleaned1["data"]
        assert "sessionUuid" not in cleaned1["data"]["nested"]

    def test_requests_dont_match_different_content(self) -> None:
        """Test that messages with different non-dynamic content don't match."""
        reader = ASLReader("dummy.asl")

        msg1 = RouteRequest()
        msg1.route.timestamp_us = 1000000

        msg2 = RouteRequest()
        msg2.route.timestamp_us = 1000001  # Different timestamp

        assert not reader.requests_match(msg1, msg2)

    def test_render_requests_match_when_dynamic_object_order_differs(self) -> None:
        """Test that render request comparison ignores dynamic object ordering."""
        reader = ASLReader("dummy.asl")

        object_a = DynamicObject(track_id="a", pose_pair=PosePair())
        object_b = DynamicObject(track_id="b", pose_pair=PosePair())

        msg1 = RGBRenderRequest(dynamic_objects=[object_a, object_b])
        msg2 = RGBRenderRequest(dynamic_objects=[object_b, object_a])

        assert reader.requests_match(msg1, msg2)

    def test_render_requests_match_with_duplicate_track_id_ordering(self) -> None:
        """Test deterministic normalization when objects share the same track_id."""
        reader = ASLReader("dummy.asl")

        object_a = DynamicObject(track_id="dup", pose_pair=PosePair())
        object_b = DynamicObject(track_id="dup", pose_pair=PosePair())
        object_a.pose_pair.start_pose.vec.x = 1.0
        object_b.pose_pair.start_pose.vec.x = 2.0

        msg1 = RGBRenderRequest(dynamic_objects=[object_a, object_b])
        msg2 = RGBRenderRequest(dynamic_objects=[object_b, object_a])

        assert reader.requests_match(msg1, msg2)


class TestFloatTolerance:
    """Test approximate float comparison in request matching."""

    def test_requests_match_tolerates_small_float_drift(self) -> None:
        """Quaternion normalization can shift float32 values by ~1 ULP.
        requests_match must tolerate this."""
        reader = ASLReader("dummy.asl")

        msg1 = RGBRenderRequest()
        msg1.sensor_pose.start_pose.quat.w = 0.5100865
        msg1.sensor_pose.start_pose.quat.x = -0.5038115
        msg1.sensor_pose.start_pose.quat.y = 0.4969628
        msg1.sensor_pose.start_pose.quat.z = -0.4888905

        msg2 = RGBRenderRequest()
        msg2.sensor_pose.start_pose.quat.w = 0.5100864
        msg2.sensor_pose.start_pose.quat.x = -0.5038114
        msg2.sensor_pose.start_pose.quat.y = 0.49696276
        msg2.sensor_pose.start_pose.quat.z = -0.48889047

        assert reader.requests_match(msg1, msg2)

    def test_requests_dont_match_large_float_difference(self) -> None:
        """Floats that differ by more than tolerance must not match."""
        reader = ASLReader("dummy.asl")

        msg1 = RGBRenderRequest()
        msg1.sensor_pose.start_pose.quat.w = 0.5
        msg1.sensor_pose.start_pose.vec.x = 1.0

        msg2 = RGBRenderRequest()
        msg2.sensor_pose.start_pose.quat.w = 0.6  # 20% different
        msg2.sensor_pose.start_pose.vec.x = 1.0

        assert not reader.requests_match(msg1, msg2)

    def test_requests_match_exact_zero(self) -> None:
        """Zero values should match exactly without division-by-zero issues."""
        reader = ASLReader("dummy.asl")

        msg1 = RGBRenderRequest()
        msg1.sensor_pose.start_pose.vec.x = 0.0

        msg2 = RGBRenderRequest()
        msg2.sensor_pose.start_pose.vec.x = 0.0

        assert reader.requests_match(msg1, msg2)

    def test_requests_match_tolerates_near_zero_float_drift(self) -> None:
        """Near-zero values can have large relative error but tiny absolute
        error.  The absolute tolerance must cover this case."""
        reader = ASLReader("dummy.asl")

        msg1 = RGBRenderRequest()
        msg1.sensor_pose.start_pose.vec.x = 5.4359436e-05

        msg2 = RGBRenderRequest()
        msg2.sensor_pose.start_pose.vec.x = 5.531311e-05  # ~1e-6 abs diff

        assert reader.requests_match(msg1, msg2)

    def test_requests_match_tolerates_accumulated_route_drift(self) -> None:
        """Route waypoints accumulate float drift across simulation steps.

        At step 12 of a 60-step replay, waypoint diffs of ~2e-6 were observed
        (e.g. -0.0003414154 vs -0.00033950806). The tolerance must accommodate
        this accumulated drift while still catching gross regressions.
        """
        reader = ASLReader("dummy.asl")

        msg1 = RouteRequest()
        msg1.route.timestamp_us = 1000000
        wp1 = msg1.route.waypoints.add()
        wp1.x = -0.0003414154
        wp1.y = -0.32271624

        msg2 = RouteRequest()
        msg2.route.timestamp_us = 1000000
        wp2 = msg2.route.waypoints.add()
        wp2.x = -0.00033950806  # ~1.9e-6 abs diff, ~0.56% rel diff
        wp2.y = -0.32271624

        assert reader.requests_match(msg1, msg2)


class TestFindingAndConsumingExchanges:
    """Test exchange finding and consumption tracking."""

    def test_find_and_consume_basic(self) -> None:
        """Test basic find and consume operation."""
        reader = ASLReader("dummy.asl")

        # Add test exchanges
        request1 = Mock(field="value1")
        response1 = Mock(result="result1")
        request2 = Mock(field="value2")
        response2 = Mock(result="result2")

        reader._add_exchange("test_service", "test_method", request1, response1)
        reader._add_exchange("test_service", "test_method", request2, response2)

        # Find and consume first request
        result = reader.find_and_consume_matching_request(
            request1, "test_service", "test_method"
        )

        assert result is not None
        assert result[0] == 0  # Index
        assert result[1] == response1
        assert 0 in reader._consumed_indices["test_service.test_method"]

    def test_consumed_messages_skipped(self) -> None:
        """Test that consumed messages are not matched again."""
        reader = ASLReader("dummy.asl")

        # Add identical requests
        request = Mock(field="same")
        reader._add_exchange("svc", "method", request, Mock(r=1))
        reader._add_exchange("svc", "method", request, Mock(r=2))
        reader._add_exchange("svc", "method", request, Mock(r=3))

        # Consume first
        result1 = reader.find_and_consume_matching_request(request, "svc", "method")
        assert result1 is not None
        assert result1[0] == 0

        # Second consumption should skip first
        result2 = reader.find_and_consume_matching_request(request, "svc", "method")
        assert result2 is not None
        assert result2[0] == 1

        # Third consumption should skip first two
        result3 = reader.find_and_consume_matching_request(request, "svc", "method")
        assert result3 is not None
        assert result3[0] == 2

    def test_lookahead_window(self) -> None:
        """Test that lookahead window limits search."""
        reader = ASLReader("dummy.asl")

        # Add 50 non-matching requests
        for i in range(50):
            reader._add_exchange("svc", "method", Mock(id=i), Mock())

        # Add 5 more non-matching
        for i in range(50, 55):
            reader._add_exchange("svc", "method", Mock(id=i), Mock())

        # Add our target at index 55
        target_request = Mock(id=999)
        reader._add_exchange("svc", "method", target_request, Mock(target=True))

        # Mock requests_match to only match our target
        def mock_match(actual: Any, expected: Any) -> bool:
            return getattr(actual, "id", None) == getattr(expected, "id", None)

        reader.requests_match = mock_match  # type: ignore[assignment]

        # Should not find request beyond lookahead window (50)
        result = reader._find_matching_request(
            Mock(id=999),
            "svc.method",
            reader._exchanges["svc.method"],
            max_lookahead=50,
        )
        assert result is None

        # Should find with larger window (need at least 56 to reach index 55)
        result = reader._find_matching_request(
            Mock(id=999),
            "svc.method",
            reader._exchanges["svc.method"],
            max_lookahead=56,
        )
        assert result is not None
        assert result[0] == 55  # Index where we added target

    def test_no_match_returns_none(self) -> None:
        """Test that find returns None when no match exists."""
        reader = ASLReader("dummy.asl")

        reader._add_exchange("svc", "method", Mock(id=1), Mock())
        reader._add_exchange("svc", "method", Mock(id=2), Mock())

        result = reader.find_and_consume_matching_request(Mock(id=999), "svc", "method")
        assert result is None

    def test_out_of_order_queries(self) -> None:
        """Test that requests can be queried out of order.

        In ASL: requestA->returnA, requestB->returnB
        Query order: requestB first, then requestA
        """
        reader = ASLReader("dummy.asl")

        # Add exchanges in ASL order: A then B
        request_a = Mock(id="A", data="request_a_data")
        response_a = Mock(id="A", result="response_a_result")
        request_b = Mock(id="B", data="request_b_data")
        response_b = Mock(id="B", result="response_b_result")

        reader._add_exchange("svc", "method", request_a, response_a)
        reader._add_exchange("svc", "method", request_b, response_b)

        # Query in reverse order: B first
        result_b = reader.find_and_consume_matching_request(request_b, "svc", "method")
        assert result_b is not None
        assert result_b[0] == 1  # Index 1 (second message)
        assert result_b[1] == response_b

        # Then query A
        result_a = reader.find_and_consume_matching_request(request_a, "svc", "method")
        assert result_a is not None
        assert result_a[0] == 0  # Index 0 (first message)
        assert result_a[1] == response_a

        # Both should be marked as consumed
        assert reader._consumed_indices["svc.method"] == {0, 1}

        # Verify complete consumption
        assert reader.is_complete()


class TestConsumptionValidation:
    """Test validation of exchange consumption completeness."""

    def test_all_consumed_no_error(self) -> None:
        """Test is_complete returns True when all exchanges consumed."""
        reader = ASLReader("dummy.asl")

        # Add and consume all exchanges
        reader._add_exchange("svc", "method", Mock(), Mock())
        reader._add_exchange("svc", "method", Mock(), Mock())
        reader._consumed_indices["svc.method"] = {0, 1}

        # Should return True
        assert reader.is_complete() is True

    def test_unconsumed_returns_false(self) -> None:
        """Test is_complete returns False with unconsumed exchanges."""
        reader = ASLReader("dummy.asl")

        # Add 5 exchanges, consume only 2
        for _ in range(5):
            reader._add_exchange("svc", "method", Mock(), Mock())
        reader._consumed_indices["svc.method"] = {0, 2}

        # Should return False
        assert reader.is_complete() is False

        # Test get_exchange_summary provides details
        summary = reader.get_exchange_summary()
        assert summary["svc.method"]["total"] == 5
        assert summary["svc.method"]["consumed"] == 2
        assert summary["svc.method"]["remaining"] == 3
        assert summary["svc.method"]["unconsumed_indices"] == [1, 3, 4]
        assert summary["svc.method"]["unconsumed_count"] == 3

    def test_is_complete(self) -> None:
        """Test is_complete() method."""
        reader = ASLReader("dummy.asl")

        # Add exchanges to multiple services
        reader._add_exchange("svc1", "method1", Mock(), Mock())
        reader._add_exchange("svc1", "method1", Mock(), Mock())
        reader._add_exchange("svc2", "method2", Mock(), Mock())

        # Not complete initially
        assert not reader.is_complete()

        # Partially consumed
        reader._consumed_indices["svc1.method1"] = {0, 1}
        assert not reader.is_complete()

        # Fully consumed
        reader._consumed_indices["svc2.method2"] = {0}
        assert reader.is_complete()


class TestSensorsimReplayEgoMasks:
    """Test that the sensorsim replay service returns ego masks from the log."""

    def test_returns_logged_ego_masks(self) -> None:
        """get_available_ego_masks returns data from the ASL exchange,
        not reconstructed from render_rgb requests."""
        from alpasim_runtime.replay_services.sensorsim_replay_service import (
            SensorsimReplayService,
        )

        reader = ASLReader("dummy.asl")

        # Set up an ego masks exchange in the reader — no render_rgb data at all.
        response = AvailableEgoMasksReturn()
        mask = response.ego_mask_metadata.add()
        mask.ego_mask_id.CopyFrom(
            EgoMaskId(
                camera_logical_id="camera_front_wide_120fov",
                rig_config_id="hyperion_8.0",
            )
        )
        request = Empty()
        reader._add_exchange("sensorsim", "get_available_ego_masks", request, response)

        service = SensorsimReplayService(reader)
        context = Mock()

        result = service.get_available_ego_masks(Empty(), context)

        assert len(result.ego_mask_metadata) == 1
        assert (
            result.ego_mask_metadata[0].ego_mask_id.camera_logical_id
            == "camera_front_wide_120fov"
        )
        assert result.ego_mask_metadata[0].ego_mask_id.rig_config_id == "hyperion_8.0"
        # Verify the exchange was consumed (proving validate_request was used)
        assert reader._consumed_indices["sensorsim.get_available_ego_masks"] == {0}


class TestCameraImageCorrelation:
    """Test camera image correlation functionality."""

    @pytest.fixture
    def reader_with_cameras(self) -> ASLReader:
        """Create reader with camera mapping and images."""
        reader = ASLReader("dummy.asl")

        # Add test images
        for camera_logical_id, timestamp in [
            ("camera_front_wide_120fov", 1000),
            ("camera_front_wide_120fov", 2000),
            ("camera_front_narrow_50fov", 1500),
            ("camera_rear_wide_120fov", 1000),
        ]:
            image = RolloutCameraImage()
            # camera_image is automatically created as a nested message
            image.camera_image.logical_id = camera_logical_id
            image.camera_image.frame_start_us = timestamp
            image.camera_image.image_bytes = (
                f"image_{camera_logical_id}_{timestamp}".encode()
            )
            reader.driver_images.append(image)

        return reader

    def test_exact_timestamp_match(self, reader_with_cameras: ASLReader) -> None:
        """Test finding image with exact timestamp."""
        result = reader_with_cameras.get_driver_image_for_camera(
            "camera_front_wide_120fov", 2000
        )
        assert result == b"image_camera_front_wide_120fov_2000"

    def test_warmup_timestamp_before_first_returns_first(
        self, reader_with_cameras: ASLReader
    ) -> None:
        """Test warmup-like timestamps before first frame return earliest image."""
        result = reader_with_cameras.get_driver_image_for_camera(
            "camera_front_wide_120fov", 900
        )
        assert result == b"image_camera_front_wide_120fov_1000"

    def test_non_exact_non_warmup_timestamp_returns_none(
        self, reader_with_cameras: ASLReader
    ) -> None:
        """Test non-warmup timestamp mismatch is treated as an error."""
        result = reader_with_cameras.get_driver_image_for_camera(
            "camera_front_wide_120fov", 1900
        )
        assert result is None

    def test_timestamp_zero_returns_first(self, reader_with_cameras: ASLReader) -> None:
        """Test that timestamp=0 returns first matching camera."""
        result = reader_with_cameras.get_driver_image_for_camera(
            "camera_front_wide_120fov", 0
        )
        # Should return first image for front wide
        assert result == b"image_camera_front_wide_120fov_1000"

    def test_no_matching_camera(self, reader_with_cameras: ASLReader) -> None:
        """Test returns None when camera not found."""
        # Remove camera 1 images
        reader_with_cameras.driver_images = [
            img
            for img in reader_with_cameras.driver_images
            if img.camera_image.logical_id != "camera_front_narrow_50fov"
        ]

        result = reader_with_cameras.get_driver_image_for_camera(
            "camera_front_narrow_50fov", 1500
        )
        assert result is None


class TestExchangeManagement:
    """Test exchange management and tracking."""

    def test_add_exchange_creates_structures(self) -> None:
        """Test that _add_exchange creates necessary data structures."""
        reader = ASLReader("dummy.asl")

        assert len(reader._exchanges) == 0
        assert len(reader._consumed_indices) == 0

        reader._add_exchange("svc", "method", Mock(), Mock())

        assert "svc.method" in reader._exchanges
        assert "svc.method" in reader._consumed_indices
        assert len(reader._exchanges["svc.method"]) == 1
        assert len(reader._consumed_indices["svc.method"]) == 0

    def test_multiple_services_tracked_separately(self) -> None:
        """Test that different services are tracked independently."""
        reader = ASLReader("dummy.asl")

        # Add to different services
        reader._add_exchange("svc1", "method", Mock(s=1), Mock())
        reader._add_exchange("svc2", "method", Mock(s=2), Mock())
        reader._add_exchange("svc1", "method", Mock(s=1), Mock())

        assert len(reader._exchanges["svc1.method"]) == 2
        assert len(reader._exchanges["svc2.method"]) == 1

        # Consume from svc1
        reader._consumed_indices["svc1.method"].add(0)

        # Check independence
        assert len(reader._consumed_indices["svc1.method"]) == 1
        assert len(reader._consumed_indices["svc2.method"]) == 0
