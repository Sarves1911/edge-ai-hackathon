from __future__ import annotations

from dataclasses import dataclass

from .config import SchedulerConfig
from .contracts import MotionResult


@dataclass(frozen=True)
class SchedulerDecision:
    run_inference: bool
    reason: str


class InferenceScheduler:
    """Choose when the expensive detector should run.

    MotionGate measures scene change. This class owns policy: startup inference,
    immediate motion response, a bounded active cadence, an after-motion hold, and
    an idle safety poll. Keeping the two responsibilities separate makes both
    components deterministic and independently testable.
    """

    def __init__(self, config: SchedulerConfig):
        self._config = config
        self.reset()

    def reset(self) -> None:
        self._frames_seen = 0
        self._last_timestamp_ms: float | None = None
        self._last_inference_ms: float | None = None
        self._active_until_ms: float | None = None

    def _record_inference(self, timestamp_ms: float, reason: str) -> SchedulerDecision:
        self._last_inference_ms = timestamp_ms
        return SchedulerDecision(run_inference=True, reason=reason)

    def _interval_elapsed(self, timestamp_ms: float, interval_ms: float) -> bool:
        return (
            self._last_inference_ms is None
            or timestamp_ms - self._last_inference_ms >= interval_ms
        )

    def decide(
        self, timestamp_ms: float, motion: MotionResult | None
    ) -> SchedulerDecision:
        if timestamp_ms < 0:
            raise ValueError("timestamp_ms cannot be negative")

        # A restarted/looped video can move its media timestamp backwards. Treat
        # that as a new stream instead of leaving scheduler deadlines in the future.
        if (
            self._last_timestamp_ms is not None
            and timestamp_ms < self._last_timestamp_ms
        ):
            self.reset()
        self._last_timestamp_ms = timestamp_ms
        self._frames_seen += 1

        if self._config.mode == "always":
            return self._record_inference(timestamp_ms, "always")

        if motion is None:
            raise ValueError("adaptive scheduling requires a motion result")

        if motion.active:
            active_until = timestamp_ms + self._config.active_hold_ms
            self._active_until_ms = max(
                active_until,
                self._active_until_ms
                if self._active_until_ms is not None
                else active_until,
            )

        if self._frames_seen <= self._config.startup_frames:
            return self._record_inference(timestamp_ms, "startup")

        # An event is an edge-triggered signal. Run immediately even if the active
        # cadence would otherwise throttle this frame.
        if motion.event:
            return self._record_inference(timestamp_ms, "motion_event")

        active_window = (
            self._active_until_ms is not None
            and timestamp_ms <= self._active_until_ms
        )
        if active_window:
            if self._interval_elapsed(
                timestamp_ms, self._config.active_interval_ms
            ):
                return self._record_inference(timestamp_ms, "active_interval")
            return SchedulerDecision(False, "active_throttle")

        if self._interval_elapsed(timestamp_ms, self._config.idle_poll_ms):
            return self._record_inference(timestamp_ms, "idle_poll")
        return SchedulerDecision(False, "idle_skip")
