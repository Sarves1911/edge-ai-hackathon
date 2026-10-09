import numpy as np

from edge_detector.config import MotionConfig
from edge_detector.motion import MotionGate


def _frame(value: int = 0) -> np.ndarray:
    return np.full((120, 160, 3), value, dtype=np.uint8)


def _warm_up(gate: MotionGate, frame: np.ndarray) -> None:
    gate.update(frame, timestamp_ms=0.0)
    gate.update(frame, timestamp_ms=33.0)
    gate.update(frame, timestamp_ms=66.0)


def test_first_frame_only_initializes_reference() -> None:
    gate = MotionGate(MotionConfig(enabled=True))
    result = gate.update(_frame(), timestamp_ms=0.0)
    assert result.score == 0.0
    assert result.changed_pixels == 0
    assert result.active is False
    assert result.event is False


def test_identical_frame_has_no_motion() -> None:
    gate = MotionGate(MotionConfig(enabled=True))
    gate.update(_frame(), timestamp_ms=0.0)
    result = gate.update(_frame(), timestamp_ms=100.0)
    assert result.active is False
    assert result.event is False


def test_large_change_triggers_motion() -> None:
    gate = MotionGate(
        MotionConfig(
            enabled=True,
            width=160,
            height=120,
            pixel_threshold=25,
            minimum_changed_area=192,
            cooldown_ms=250.0,
        )
    )
    first = _frame()
    second = _frame()
    second[20:60, 20:60] = 255

    _warm_up(gate, first)
    result = gate.update(second, timestamp_ms=100.0)

    assert result.changed_pixels >= 1_500
    assert result.active is True
    assert result.event is True


def test_small_change_stays_below_area_threshold() -> None:
    gate = MotionGate(MotionConfig(enabled=True, minimum_changed_area=192))
    first = _frame()
    second = _frame()
    second[0:5, 0:5] = 255

    _warm_up(gate, first)
    result = gate.update(second, timestamp_ms=100.0)

    assert result.changed_pixels < 192
    assert result.active is False
    assert result.event is False


def test_cooldown_debounces_events_but_not_active_motion() -> None:
    gate = MotionGate(MotionConfig(enabled=True, cooldown_ms=250.0))
    dark = _frame(0)
    bright = _frame(255)

    _warm_up(gate, dark)
    first_event = gate.update(bright, timestamp_ms=100.0)
    during_cooldown = gate.update(dark, timestamp_ms=200.0)
    after_cooldown = gate.update(bright, timestamp_ms=400.0)

    assert first_event.event is True
    assert during_cooldown.active is True
    assert during_cooldown.event is False
    assert after_cooldown.event is True
