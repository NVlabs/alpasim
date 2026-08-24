# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor

import navsim_diffusiondrive_challenge.batch_worker as batch_worker_module
import numpy as np
import pytest
from navsim_diffusiondrive_challenge.batch_worker import BatchWorker
from navsim_diffusiondrive_challenge.policy import InferenceInput, Prediction
from navsim_diffusiondrive_challenge.preprocessing import CAMERA_IDS


def _request(marker: float) -> InferenceInput:
    return InferenceInput(
        images={
            camera_id: np.zeros((1, 1, 3), dtype=np.uint8) for camera_id in CAMERA_IDS
        },
        command_one_hot=np.array([0, 1, 0, 0], dtype=np.float32),
        velocity_xy=np.array([marker, 0], dtype=np.float32),
        acceleration_xy=np.zeros(2, dtype=np.float32),
    )


def _prediction(marker: float) -> Prediction:
    return Prediction(
        trajectory=np.full((8, 3), marker, dtype=np.float32),
    )


def _submit_concurrently(
    executor: ThreadPoolExecutor,
    worker: BatchWorker,
    requests: list[InferenceInput],
) -> list[Future[Prediction]]:
    barrier = threading.Barrier(len(requests) + 1)

    def predict(request: InferenceInput) -> Prediction:
        barrier.wait(timeout=1.0)
        return worker.predict(request, timeout=2.0)

    futures = [executor.submit(predict, request) for request in requests]
    barrier.wait(timeout=1.0)
    return futures


class RecordingPolicy:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self._lock = threading.Lock()

    def predict_batch(self, requests: list[InferenceInput]) -> list[Prediction]:
        with self._lock:
            self.batch_sizes.append(len(requests))
        return [_prediction(float(request.velocity_xy[0])) for request in requests]


def test_two_concurrent_requests_share_one_forward() -> None:
    policy = RecordingPolicy()
    worker = BatchWorker(policy, max_batch_size=2, batch_window_s=0.2)
    worker.start()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = _submit_concurrently(
                executor,
                worker,
                [_request(11), _request(22)],
            )
            results = [future.result(timeout=3.0) for future in futures]

        assert policy.batch_sizes == [2]
        assert [result.trajectory[0, 0] for result in results] == [11.0, 22.0]
    finally:
        worker.stop()


def test_single_request_runs_after_batch_window() -> None:
    policy = RecordingPolicy()
    worker = BatchWorker(policy, max_batch_size=2, batch_window_s=0.01)
    worker.start()
    try:
        result = worker.predict(_request(7), timeout=1.0)

        assert policy.batch_sizes == [1]
        assert result.trajectory[0, 0] == 7.0
    finally:
        worker.stop()


def test_policy_exception_reaches_every_request_in_batch() -> None:
    class FailingPolicy:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def predict_batch(
            self,
            requests: list[InferenceInput],
        ) -> list[Prediction]:
            self.batch_sizes.append(len(requests))
            raise ValueError("inference failed")

    policy = FailingPolicy()
    worker = BatchWorker(policy, max_batch_size=2, batch_window_s=0.2)
    worker.start()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = _submit_concurrently(
                executor,
                worker,
                [_request(1), _request(2)],
            )
            for future in futures:
                with pytest.raises(ValueError, match="^inference failed$"):
                    future.result(timeout=3.0)
        assert policy.batch_sizes == [2]
    finally:
        worker.stop()


def test_wrong_policy_output_length_reaches_every_request_in_batch() -> None:
    class WrongLengthPolicy:
        def predict_batch(
            self,
            requests: list[InferenceInput],
        ) -> list[Prediction]:
            return [_prediction(1)]

    worker = BatchWorker(WrongLengthPolicy(), max_batch_size=2, batch_window_s=0.2)
    worker.start()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = _submit_concurrently(
                executor,
                worker,
                [_request(1), _request(2)],
            )
            for future in futures:
                with pytest.raises(
                    RuntimeError,
                    match="^policy returned the wrong batch size$",
                ):
                    future.result(timeout=3.0)
    finally:
        worker.stop()


def test_start_and_stop_are_idempotent_and_leave_no_thread() -> None:
    worker = BatchWorker(RecordingPolicy())

    worker.start()
    thread = worker._thread
    worker.start()

    assert thread is not None
    assert worker._thread is thread
    assert thread.name == "diffusiondrive-batch-worker"
    assert thread.daemon

    worker.stop()
    worker.stop()

    assert worker._thread is None
    assert not thread.is_alive()


def test_stop_timeout_preserves_worker_until_blocked_policy_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingPolicy(RecordingPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def predict_batch(
            self,
            requests: list[InferenceInput],
        ) -> list[Prediction]:
            self.entered.set()
            if not self.release.wait(timeout=0.25):
                raise TimeoutError("test policy was not released")
            return super().predict_batch(requests)

    monkeypatch.setattr(
        batch_worker_module,
        "_STOP_TIMEOUT_S",
        0.02,
        raising=False,
    )
    policy = BlockingPolicy()
    worker = BatchWorker(policy, batch_window_s=0)
    worker.start()
    original_thread = worker._thread
    assert original_thread is not None

    with ThreadPoolExecutor(max_workers=1) as executor:
        prediction_future = executor.submit(worker.predict, _request(1), 1.0)
        assert policy.entered.wait(timeout=1.0)
        try:
            with pytest.raises(RuntimeError, match="batch worker did not stop"):
                worker.stop()

            assert worker._thread is original_thread
            worker.start()
            assert worker._thread is original_thread
            with pytest.raises(RuntimeError, match="batch worker is stopping"):
                worker.predict(_request(2), timeout=0.1)
        finally:
            policy.release.set()

        assert prediction_future.result(timeout=1.0).trajectory[0, 0] == 1.0

    original_thread.join(timeout=1.0)
    assert not original_thread.is_alive()
    worker.stop()
    assert worker._thread is None

    worker.start()
    restarted_thread = worker._thread
    assert restarted_thread is not None
    assert restarted_thread is not original_thread
    try:
        assert worker.predict(_request(3), timeout=1.0).trajectory[0, 0] == 3.0
    finally:
        worker.stop()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_batch_size": 0}, "max_batch_size must be a positive integer"),
        ({"max_batch_size": -1}, "max_batch_size must be a positive integer"),
        ({"max_batch_size": 1.5}, "max_batch_size must be a positive integer"),
        ({"max_batch_size": True}, "max_batch_size must be a positive integer"),
        ({"batch_window_s": -0.001}, "batch_window_s must be non-negative"),
        ({"batch_window_s": float("nan")}, "batch_window_s must be non-negative"),
        ({"batch_window_s": float("inf")}, "batch_window_s must be non-negative"),
        ({"batch_window_s": True}, "batch_window_s must be non-negative"),
    ],
)
def test_constructor_rejects_invalid_batch_settings(
    kwargs: dict[str, int | float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=f"^{message}$"):
        BatchWorker(RecordingPolicy(), **kwargs)


def test_predict_requires_running_worker() -> None:
    worker = BatchWorker(RecordingPolicy())

    with pytest.raises(RuntimeError, match="^batch worker is not running$"):
        worker.predict(_request(1), timeout=0.1)

    worker.start()
    worker.stop()

    with pytest.raises(RuntimeError, match="^batch worker is not running$"):
        worker.predict(_request(2), timeout=0.1)
