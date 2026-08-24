# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Protocol

from .policy import InferenceInput, Prediction

LOGGER = logging.getLogger(__name__)
_STOP_TIMEOUT_S = 5.0


class BatchPolicy(Protocol):
    def predict_batch(self, requests: list[InferenceInput]) -> list[Prediction]: ...


@dataclass
class _Pending:
    request: InferenceInput
    future: Future[Prediction]


class BatchWorker:
    def __init__(
        self,
        policy: BatchPolicy,
        *,
        max_batch_size: int = 2,
        batch_window_s: float = 0.002,
    ) -> None:
        if (
            isinstance(max_batch_size, bool)
            or not isinstance(max_batch_size, Integral)
            or max_batch_size <= 0
        ):
            raise ValueError("max_batch_size must be a positive integer")
        if (
            isinstance(batch_window_s, bool)
            or not isinstance(batch_window_s, Real)
            or not math.isfinite(batch_window_s)
            or batch_window_s < 0
        ):
            raise ValueError("batch_window_s must be non-negative")

        self._policy = policy
        self._max_batch_size = int(max_batch_size)
        self._batch_window_s = float(batch_window_s)
        self._queue: queue.Queue[_Pending | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._lifecycle_lock = threading.Lock()

    def start(self) -> None:
        with self._lifecycle_lock:
            self._reap_stopped_thread_locked()
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run,
                name="ltf-batch-worker",
                daemon=True,
            )
            self._thread.start()

    def predict(
        self,
        request: InferenceInput,
        timeout: float | None = None,
    ) -> Prediction:
        future: Future[Prediction] = Future()
        with self._lifecycle_lock:
            self._reap_stopped_thread_locked()
            if self._stopping:
                raise RuntimeError("batch worker is stopping")
            if self._thread is None:
                raise RuntimeError("batch worker is not running")
            self._queue.put(_Pending(request=request, future=future))
        return future.result(timeout=timeout)

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._reap_stopped_thread_locked()
            thread = self._thread
            if thread is None:
                return
            if not self._stopping:
                self._stopping = True
                self._queue.put(None)

        thread.join(timeout=_STOP_TIMEOUT_S)
        if thread.is_alive():
            raise RuntimeError(
                f"batch worker did not stop within {_STOP_TIMEOUT_S:g} seconds"
            )

        with self._lifecycle_lock:
            if self._thread is thread:
                self._thread = None
                self._stopping = False

    def _reap_stopped_thread_locked(self) -> None:
        if self._thread is not None and not self._thread.is_alive():
            self._thread = None
            self._stopping = False

    def _run(self) -> None:
        while True:
            first = self._queue.get()
            if first is None:
                return

            batch = [first]
            deadline = time.monotonic() + self._batch_window_s
            while len(batch) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    pending = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if pending is None:
                    self._queue.put(None)
                    break
                batch.append(pending)

            try:
                LOGGER.info("ltf_batch_size=%d", len(batch))
                outputs = self._policy.predict_batch(
                    [pending.request for pending in batch]
                )
                if len(outputs) != len(batch):
                    raise RuntimeError("policy returned the wrong batch size")
            except BaseException as exc:
                for pending in batch:
                    pending.future.set_exception(exc)
            else:
                for pending, output in zip(batch, outputs, strict=True):
                    pending.future.set_result(output)
