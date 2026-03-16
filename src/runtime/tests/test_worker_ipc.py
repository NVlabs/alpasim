# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Unit tests for worker IPC types.
"""

import pickle
from multiprocessing import Queue
from unittest.mock import MagicMock

from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_runtime.address_pool import ServiceAddress
from alpasim_runtime.worker.ipc import (
    SHUTDOWN_SENTINEL,
    AssignedRolloutJob,
    JobResult,
    PendingRolloutJob,
    ServiceEndpoints,
    WorkerArgs,
    _ShutdownSentinel,
)


def _make_endpoints() -> ServiceEndpoints:
    """Create a test ServiceEndpoints instance."""
    return ServiceEndpoints(
        driver=ServiceAddress("driver:50051", skip=False),
        sensorsim=ServiceAddress("sensorsim:50052", skip=False),
        physics=ServiceAddress("physics:50053", skip=False),
        trafficsim=ServiceAddress("trafficsim:50054", skip=False),
        controller=ServiceAddress("controller:50055", skip=False),
    )


class TestPendingRolloutJob:
    """Tests for PendingRolloutJob dataclass."""

    def test_creation(self):
        """Pending jobs contain only identity and scene."""
        job = PendingRolloutJob(
            job_id="test-123",
            scene_id="test-scene",
            rollout_spec_index=2,
            artifact_path="/tmp/test-scene.usdz",
        )

        assert job.job_id == "test-123"
        assert job.scene_id == "test-scene"
        assert job.rollout_spec_index == 2
        assert job.artifact_path == "/tmp/test-scene.usdz"

    def test_pickling(self):
        """PendingRolloutJob should be picklable."""
        job = PendingRolloutJob(
            job_id="test-123",
            scene_id="test-scene",
            rollout_spec_index=1,
            artifact_path="/tmp/test-scene.usdz",
        )
        pickled = pickle.dumps(job)
        unpickled = pickle.loads(pickled)

        assert unpickled.job_id == job.job_id
        assert unpickled.scene_id == job.scene_id
        assert unpickled.rollout_spec_index == job.rollout_spec_index
        assert unpickled.artifact_path == job.artifact_path


class TestAssignedRolloutJob:
    """Tests for AssignedRolloutJob dataclass."""

    def test_creation(self):
        """Assigned jobs include artifact path and service endpoints."""
        ep = _make_endpoints()
        job = AssignedRolloutJob(
            request_id="req-1",
            job_id="test-123",
            scene_id="test-scene",
            rollout_spec_index=0,
            artifact_path="/tmp/test-scene.usdz",
            endpoints=ep,
        )
        assert job.artifact_path == "/tmp/test-scene.usdz"
        assert job.endpoints.driver.address == "driver:50051"

    def test_pickling(self):
        """AssignedRolloutJob should be picklable for multiprocessing Queue."""
        ep = _make_endpoints()
        job = AssignedRolloutJob(
            request_id="req-1",
            job_id="test-123",
            scene_id="test-scene",
            rollout_spec_index=0,
            artifact_path="/tmp/test-scene.usdz",
            endpoints=ep,
        )

        pickled = pickle.dumps(job)
        unpickled = pickle.loads(pickled)

        assert unpickled.job_id == job.job_id
        assert unpickled.artifact_path == "/tmp/test-scene.usdz"
        assert unpickled.endpoints.driver.address == "driver:50051"
        assert unpickled.endpoints.physics.skip is False


class TestServiceEndpoints:
    """Tests for ServiceEndpoints dataclass."""

    def test_creation(self):
        ep = _make_endpoints()
        assert ep.driver.address == "driver:50051"
        assert ep.sensorsim.address == "sensorsim:50052"
        assert ep.physics.address == "physics:50053"
        assert ep.trafficsim.address == "trafficsim:50054"
        assert ep.controller.address == "controller:50055"

    def test_with_skip(self):
        ep = ServiceEndpoints(
            driver=ServiceAddress("skip", skip=True),
            sensorsim=ServiceAddress("addr:1", skip=False),
            physics=ServiceAddress("skip", skip=True),
            trafficsim=ServiceAddress("addr:2", skip=False),
            controller=ServiceAddress("skip", skip=True),
        )
        assert ep.driver.skip is True
        assert ep.sensorsim.skip is False

    def test_pickling(self):
        ep = _make_endpoints()
        pickled = pickle.dumps(ep)
        unpickled = pickle.loads(pickled)
        assert unpickled.driver.address == ep.driver.address


class TestJobResult:
    """Tests for JobResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = JobResult(
            request_id="req-1",
            job_id="test-123",
            rollout_spec_index=0,
            success=True,
            error=None,
            error_traceback=None,
            rollout_uuid="uuid-456",
        )

        assert result.success is True
        assert result.error is None
        assert result.rollout_uuid == "uuid-456"

    def test_failure_result(self):
        """Test failure result."""
        result = JobResult(
            request_id="req-1",
            job_id="test-123",
            rollout_spec_index=0,
            success=False,
            error="Something went wrong",
            error_traceback="Traceback...",
            rollout_uuid=None,
        )

        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.error_traceback == "Traceback..."

    def test_pickling(self):
        """JobResult should be picklable for multiprocessing Queue."""
        result = JobResult(
            request_id="req-1",
            job_id="test-123",
            rollout_spec_index=0,
            success=False,
            error="Error message",
            error_traceback="Full traceback",
            rollout_uuid="uuid-789",
        )

        pickled = pickle.dumps(result)
        unpickled = pickle.loads(pickled)

        assert unpickled.job_id == result.job_id
        assert unpickled.success == result.success
        assert unpickled.error == result.error
        assert unpickled.request_id == result.request_id


class TestShutdownSentinel:
    """Tests for shutdown sentinel."""

    def test_singleton_identity(self):
        """SHUTDOWN_SENTINEL should be a singleton-like object."""
        assert isinstance(SHUTDOWN_SENTINEL, _ShutdownSentinel)

    def test_distinct_from_none(self):
        """Sentinel should be distinct from None."""
        assert SHUTDOWN_SENTINEL is not None

    def test_pickling(self):
        """Sentinel should be picklable."""
        pickled = pickle.dumps(SHUTDOWN_SENTINEL)
        unpickled = pickle.loads(pickled)
        # After unpickling it's a new instance but same type
        assert isinstance(unpickled, _ShutdownSentinel)


class TestWorkerArgs:
    """Tests for WorkerArgs dataclass."""

    def test_creation_with_version_ids(self):
        """WorkerArgs should include version_ids."""
        from alpasim_grpc.v0.common_pb2 import VersionId

        version_ids = RolloutMetadata.VersionIds(
            runtime_version=VersionId(version_id="0.3.0", git_hash="abc"),
        )
        args = WorkerArgs(
            worker_id=0,
            num_workers=2,
            job_queue=Queue(),
            result_queue=Queue(),
            num_consumers=4,
            user_config_path="/tmp/config.yaml",
            log_dir="/tmp/logs",
            eval_config=MagicMock(),
            version_ids=version_ids,
        )
        assert args.version_ids is version_ids
        assert args.worker_id == 0
        assert args.num_consumers == 4

    def test_version_ids_field_round_trips(self):
        """version_ids protobuf should survive serialization (as it would through multiprocessing)."""
        from alpasim_grpc.v0.common_pb2 import VersionId

        version_ids = RolloutMetadata.VersionIds(
            runtime_version=VersionId(version_id="0.3.0", git_hash="abc"),
            egodriver_version=VersionId(version_id="1.0.0", git_hash="def"),
        )

        # Protobuf messages are picklable by default
        pickled = pickle.dumps(version_ids)
        unpickled = pickle.loads(pickled)

        assert unpickled.runtime_version.version_id == "0.3.0"
        assert unpickled.egodriver_version.version_id == "1.0.0"
