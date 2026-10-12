from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Condition, Thread
from time import perf_counter, sleep
from typing import Any

from .backend import Detector
from .contracts import Detection


@dataclass(frozen=True)
class DetectorJob:
    """One owned frame submitted to the detector worker."""

    frame_index: int
    timestamp_ms: float
    image: Any
    reason: str
    submitted_at_monotonic_ms: float

    @classmethod
    def from_frame(
        cls,
        *,
        frame_index: int,
        timestamp_ms: float,
        image: Any,
        reason: str,
    ) -> DetectorJob:
        # Capture can reuse backend-owned memory. The worker therefore owns a copy
        # instead of retaining a view into the camera or decoder buffer.
        owned_image = image.copy()
        return cls(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            image=owned_image,
            reason=reason,
            submitted_at_monotonic_ms=perf_counter() * 1_000.0,
        )


@dataclass(frozen=True)
class DetectorResult:
    frame_index: int
    timestamp_ms: float
    image: Any
    reason: str
    submitted_at_monotonic_ms: float
    started_at_monotonic_ms: float
    completed_at_monotonic_ms: float
    detections: tuple[Detection, ...]
    detector_ms: float
    error: Exception | None = None

    @property
    def queue_wait_ms(self) -> float:
        return max(0.0, self.started_at_monotonic_ms - self.submitted_at_monotonic_ms)


class AsyncDetectorWorker:
    """Single detector thread with one replaceable pending job.

    There is never a FIFO backlog: one job may be running and at most one newer
    job may wait. Publishing again replaces that pending job, so overload favors
    the freshest frame.
    """

    def __init__(
        self,
        detector: Detector,
        *,
        simulated_delay_ms: float = 0.0,
        sleep_fn: Callable[[float], None] = sleep,
    ) -> None:
        self._detector = detector
        self._simulated_delay_ms = simulated_delay_ms
        self._sleep = sleep_fn
        self._condition = Condition()
        self._pending: DetectorJob | None = None
        self._result: DetectorResult | None = None
        self._thread: Thread | None = None
        self._running = False
        self._stop_requested = False
        self._drain_on_stop = False

        self._submitted = 0
        self._started = 0
        self._completed = 0
        self._replaced = 0
        self._dropped_results = 0
        self._discarded_on_stop = 0
        self._detector_durations_ms: list[float] = []
        self._detection_count = 0

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                return
            if self._stop_requested:
                raise RuntimeError("A stopped detector worker cannot be restarted")
            self._thread = Thread(
                target=self._run,
                name="edge-detector-worker",
                daemon=False,
            )
            self._thread.start()

    def submit(self, job: DetectorJob) -> bool:
        with self._condition:
            if self._thread is None:
                raise RuntimeError("Detector worker must be started before submit")
            if self._stop_requested:
                return False
            self._submitted += 1
            if self._pending is not None:
                self._replaced += 1
            self._pending = job
            self._condition.notify_all()
            return True

    def take_result(
        self,
        *,
        wait: bool = False,
        timeout_s: float | None = None,
    ) -> DetectorResult | None:
        with self._condition:
            if wait:
                ready = self._condition.wait_for(
                    lambda: self._result is not None
                    or (
                        self._stop_requested
                        and not self._running
                        and self._pending is None
                    ),
                    timeout=timeout_s,
                )
                if not ready:
                    return None
            result = self._result
            self._result = None
            return result

    def stop(self, *, drain: bool = False, timeout_s: float | None = None) -> bool:
        with self._condition:
            if self._thread is None:
                return True
            self._stop_requested = True
            self._drain_on_stop = drain
            if not drain and self._pending is not None:
                self._pending = None
                self._discarded_on_stop += 1
            thread = self._thread
            self._condition.notify_all()
        thread.join(timeout=timeout_s)
        return not thread.is_alive()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._pending is not None or self._stop_requested
                )
                if self._stop_requested and (
                    self._pending is None or not self._drain_on_stop
                ):
                    self._running = False
                    self._condition.notify_all()
                    return
                job = self._pending
                self._pending = None
                if job is None:
                    continue
                self._running = True
                self._started += 1
                started_at_ms = perf_counter() * 1_000.0
                self._condition.notify_all()

            error: Exception | None = None
            detections: tuple[Detection, ...] = ()
            detector_ms = 0.0
            try:
                if self._simulated_delay_ms > 0.0:
                    self._sleep(self._simulated_delay_ms / 1_000.0)
                detections, detector_ms = self._detector.detect(job.image)
            except Exception as caught:  # noqa: BLE001 - surfaced on the main thread
                error = caught
            completed_at_ms = perf_counter() * 1_000.0
            result = DetectorResult(
                frame_index=job.frame_index,
                timestamp_ms=job.timestamp_ms,
                image=job.image,
                reason=job.reason,
                submitted_at_monotonic_ms=job.submitted_at_monotonic_ms,
                started_at_monotonic_ms=started_at_ms,
                completed_at_monotonic_ms=completed_at_ms,
                detections=detections,
                detector_ms=detector_ms,
                error=error,
            )

            with self._condition:
                self._running = False
                self._completed += 1
                if error is None:
                    self._detector_durations_ms.append(detector_ms)
                    self._detection_count += len(detections)
                if self._result is not None:
                    self._dropped_results += 1
                self._result = result
                self._condition.notify_all()

    @property
    def is_alive(self) -> bool:
        with self._condition:
            return self._thread is not None and self._thread.is_alive()

    def stats(self) -> dict[str, int | bool]:
        with self._condition:
            return {
                "submitted": self._submitted,
                "started": self._started,
                "completed": self._completed,
                "replaced_pending": self._replaced,
                "dropped_results": self._dropped_results,
                "discarded_on_stop": self._discarded_on_stop,
                "pending": 0 if self._pending is None else 1,
                "running": self._running,
                "max_pending": 1,
            }

    def detector_observations(self) -> tuple[list[float], int]:
        with self._condition:
            return list(self._detector_durations_ms), self._detection_count
