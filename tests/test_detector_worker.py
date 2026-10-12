from __future__ import annotations

from threading import Event, Lock, get_ident
from time import perf_counter

import numpy as np
import pytest

from edge_detector.detector_worker import AsyncDetectorWorker, DetectorJob


class _RecordingDetector:
    def __init__(self) -> None:
        self.thread_ids: list[int] = []
        self.frames: list[int] = []
        self._lock = Lock()

    def detect(self, frame: np.ndarray):
        with self._lock:
            self.thread_ids.append(get_ident())
            self.frames.append(int(frame[0, 0, 0]))
        return (), 1.0


class _BlockingDetector(_RecordingDetector):
    def __init__(self) -> None:
        super().__init__()
        self.entered = Event()
        self.release = Event()

    def detect(self, frame: np.ndarray):
        with self._lock:
            self.thread_ids.append(get_ident())
            self.frames.append(int(frame[0, 0, 0]))
        if len(self.frames) == 1:
            self.entered.set()
            assert self.release.wait(timeout=2.0)
        return (), 1.0


def _job(index: int) -> DetectorJob:
    return DetectorJob.from_frame(
        frame_index=index,
        timestamp_ms=index * 33.0,
        image=np.full((4, 4, 3), index, dtype=np.uint8),
        reason="test",
    )


def test_detector_runs_on_worker_thread() -> None:
    detector = _RecordingDetector()
    worker = AsyncDetectorWorker(detector)
    worker.start()

    assert worker.submit(_job(1)) is True
    result = worker.take_result(wait=True, timeout_s=1.0)

    assert result is not None
    assert result.frame_index == 1
    assert len(detector.thread_ids) == 1
    assert detector.thread_ids[0] != get_ident()
    assert worker.stop(timeout_s=1.0) is True


def test_one_pending_slot_and_latest_frame_wins() -> None:
    detector = _BlockingDetector()
    worker = AsyncDetectorWorker(detector)
    worker.start()

    worker.submit(_job(0))
    assert detector.entered.wait(timeout=1.0)
    worker.submit(_job(1))
    worker.submit(_job(2))
    worker.submit(_job(3))

    overloaded = worker.stats()
    assert overloaded["pending"] == 1
    assert overloaded["max_pending"] == 1
    assert overloaded["replaced_pending"] == 2

    detector.release.set()
    assert worker.stop(drain=True, timeout_s=2.0) is True
    assert detector.frames == [0, 3]
    assert worker.stats()["completed"] == 2


def test_worker_shuts_down_cleanly_and_rejects_new_work() -> None:
    worker = AsyncDetectorWorker(_RecordingDetector())
    worker.start()

    assert worker.stop(timeout_s=1.0) is True
    assert worker.is_alive is False
    assert worker.submit(_job(1)) is False


@pytest.mark.parametrize("delay_ms", [100.0, 200.0, 300.0, 500.0])
def test_simulated_detector_latency_does_not_block_submit(delay_ms: float) -> None:
    detector = _RecordingDetector()
    worker = AsyncDetectorWorker(detector, simulated_delay_ms=delay_ms)
    worker.start()

    submitted_at = perf_counter()
    assert worker.submit(_job(1)) is True
    submit_elapsed_ms = (perf_counter() - submitted_at) * 1_000.0
    result = worker.take_result(wait=True, timeout_s=2.0)

    assert submit_elapsed_ms < 50.0
    assert result is not None
    worker_elapsed_ms = (
        result.completed_at_monotonic_ms - result.started_at_monotonic_ms
    )
    assert worker_elapsed_ms >= delay_ms * 0.9
    assert worker.stop(timeout_s=1.0) is True
