from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from .config import MotionConfig
from .contracts import MotionResult


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is not installed. Run `uv sync` from the project root."
        ) from error
    return cv2


class MotionGate:
    """Cheap frame-difference motion signal for the later adaptive scheduler.

    The gate deliberately does not decide whether neural inference runs. It only
    converts consecutive frames into a small, deterministic motion signal.

    The first few frames are treated as a startup warm-up period so camera
    exposure/decoder transients do not produce false motion events.
    """

    _WARMUP_FRAMES = 3

    def __init__(self, config: MotionConfig):
        self._config = config
        self._previous_gray: Any | None = None
        self._last_event_ms: float | None = None
        self._frames_seen = 0

    def reset(self) -> None:
        self._previous_gray = None
        self._last_event_ms = None
        self._frames_seen = 0

    def update(self, frame: Any, timestamp_ms: float) -> MotionResult:
        cv2 = _cv2()
        started = perf_counter()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.resize(
            gray,
            (self._config.width, self._config.height),
            interpolation=cv2.INTER_AREA,
        )

        # Frame 1 only establishes the reference image. There is no previous
        # frame available, so motion cannot be measured yet.
        if self._previous_gray is None:
            self._previous_gray = gray
            self._frames_seen += 1

            return MotionResult(
                score=0.0,
                changed_pixels=0,
                active=False,
                event=False,
                processing_ms=(perf_counter() - started) * 1_000.0,
            )

        self._frames_seen += 1

        # Compare the current grayscale frame with the immediately previous one.
        delta = cv2.absdiff(gray, self._previous_gray)

        changed_pixels = int(
            np.count_nonzero(
                delta >= self._config.pixel_threshold
            )
        )

        total_pixels = (
            self._config.width
            * self._config.height
        )

        score = changed_pixels / total_pixels

        active = (
            changed_pixels
            >= self._config.minimum_changed_area
        )

        # Suppress startup motion decisions while still measuring and updating
        # the reference frame. This prevents camera exposure/decoder startup
        # transients from triggering a false semantic event.
        if self._frames_seen <= self._WARMUP_FRAMES:
            self._previous_gray = gray

            return MotionResult(
                score=float(score),
                changed_pixels=changed_pixels,
                active=False,
                event=False,
                processing_ms=(perf_counter() - started) * 1_000.0,
            )

        cooldown_elapsed = (
            self._last_event_ms is None
            or timestamp_ms < self._last_event_ms
            or (
                timestamp_ms - self._last_event_ms
                >= self._config.cooldown_ms
            )
        )

        event = active and cooldown_elapsed

        if event:
            self._last_event_ms = timestamp_ms

        # Always compare adjacent frames. We are measuring scene change rather
        # than distance from an old keyframe.
        self._previous_gray = gray

        return MotionResult(
            score=float(score),
            changed_pixels=changed_pixels,
            active=active,
            event=event,
            processing_ms=(perf_counter() - started) * 1_000.0,
        )