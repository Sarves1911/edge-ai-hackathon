import pytest

from edge_detector.config import SchedulerConfig
from edge_detector.contracts import MotionResult
from edge_detector.scheduler import InferenceScheduler


def _motion(*, active: bool = False, event: bool = False) -> MotionResult:
    return MotionResult(
        score=0.1 if active else 0.0,
        changed_pixels=1_920 if active else 0,
        active=active,
        event=event,
        processing_ms=0.1,
    )


def test_always_mode_runs_every_frame_without_motion() -> None:
    scheduler = InferenceScheduler(SchedulerConfig(mode="always"))

    decisions = [scheduler.decide(timestamp, None) for timestamp in (0.0, 33.0, 66.0)]

    assert all(decision.run_inference for decision in decisions)
    assert {decision.reason for decision in decisions} == {"always"}


def test_adaptive_mode_runs_startup_then_idle_poll() -> None:
    scheduler = InferenceScheduler(
        SchedulerConfig(mode="adaptive", startup_frames=2, idle_poll_ms=1_000.0)
    )

    first = scheduler.decide(0.0, _motion())
    second = scheduler.decide(33.0, _motion())
    skipped = scheduler.decide(66.0, _motion())
    poll = scheduler.decide(1_033.0, _motion())

    assert (first.run_inference, first.reason) == (True, "startup")
    assert (second.run_inference, second.reason) == (True, "startup")
    assert (skipped.run_inference, skipped.reason) == (False, "idle_skip")
    assert (poll.run_inference, poll.reason) == (True, "idle_poll")


def test_motion_event_starts_active_cadence_and_hold() -> None:
    scheduler = InferenceScheduler(
        SchedulerConfig(
            mode="adaptive",
            startup_frames=1,
            active_interval_ms=100.0,
            active_hold_ms=300.0,
            idle_poll_ms=1_000.0,
        )
    )

    assert scheduler.decide(0.0, _motion()).reason == "startup"
    event = scheduler.decide(20.0, _motion(active=True, event=True))
    throttled = scheduler.decide(50.0, _motion(active=True))
    cadence = scheduler.decide(120.0, _motion())
    after_hold = scheduler.decide(400.0, _motion())
    poll = scheduler.decide(1_120.0, _motion())

    assert (event.run_inference, event.reason) == (True, "motion_event")
    assert (throttled.run_inference, throttled.reason) == (
        False,
        "active_throttle",
    )
    assert (cadence.run_inference, cadence.reason) == (True, "active_interval")
    assert (after_hold.run_inference, after_hold.reason) == (False, "idle_skip")
    assert (poll.run_inference, poll.reason) == (True, "idle_poll")


def test_adaptive_mode_requires_motion_result() -> None:
    scheduler = InferenceScheduler(
        SchedulerConfig(mode="adaptive", startup_frames=1)
    )

    with pytest.raises(ValueError, match="motion result"):
        scheduler.decide(0.0, None)


def test_timestamp_regression_restarts_startup_sequence() -> None:
    scheduler = InferenceScheduler(
        SchedulerConfig(mode="adaptive", startup_frames=1)
    )
    scheduler.decide(100.0, _motion())

    restarted = scheduler.decide(0.0, _motion())

    assert (restarted.run_inference, restarted.reason) == (True, "startup")
