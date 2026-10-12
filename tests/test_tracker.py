from __future__ import annotations

import numpy as np

from edge_detector.config import TrackerConfig
from edge_detector.contracts import Detection
from edge_detector.tracker import OpticalFlowTracker


class _FakeCv2:
    COLOR_BGR2GRAY = 1
    TERM_CRITERIA_EPS = 2
    TERM_CRITERIA_COUNT = 4

    def __init__(self, *, dx: float = 0.0, dy: float = 0.0, fail: bool = False):
        self.dx = dx
        self.dy = dy
        self.fail = fail

    @staticmethod
    def cvtColor(frame: np.ndarray, _: int) -> np.ndarray:
        return frame[:, :, 0]

    @staticmethod
    def goodFeaturesToTrack(
        roi: np.ndarray,
        *,
        maxCorners: int,
        qualityLevel: float,
        minDistance: float,
        blockSize: int,
    ) -> np.ndarray:
        del roi, maxCorners, qualityLevel, minDistance, blockSize
        return np.array(
            [[[2.0, 2.0]], [[6.0, 2.0]], [[2.0, 6.0]], [[6.0, 6.0]]],
            dtype=np.float32,
        )

    def calcOpticalFlowPyrLK(
        self,
        previous: np.ndarray,
        current: np.ndarray,
        points: np.ndarray,
        _: None,
        **kwargs: object,
    ):
        del previous, current, kwargs
        shifted = points + np.array([self.dx, self.dy], dtype=np.float32)
        status_value = 0 if self.fail else 1
        status = np.full((len(points), 1), status_value, dtype=np.uint8)
        return shifted, status, np.zeros((len(points), 1), dtype=np.float32)


def _person() -> Detection:
    return Detection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        x1=10.0,
        y1=20.0,
        x2=30.0,
        y2=50.0,
    )


def _frame() -> np.ndarray:
    return np.zeros((80, 100, 3), dtype=np.uint8)


def test_tracker_initializes_only_from_yolo_detections() -> None:
    tracker = OpticalFlowTracker(TrackerConfig(), cv2_module=_FakeCv2())

    update = tracker.initialize(
        _frame(), (_person(),), frame_index=4, timestamp_ms=132.0
    )

    assert update.detections == (_person(),)
    assert update.tracked is False
    assert tracker.frame_index == 4
    assert tracker.stats()["initializations"] == 1


def test_tracker_updates_boxes_between_yolo_results() -> None:
    tracker = OpticalFlowTracker(
        TrackerConfig(), cv2_module=_FakeCv2(dx=3.0, dy=-2.0)
    )
    tracker.initialize(_frame(), (_person(),), frame_index=1, timestamp_ms=0.0)

    update = tracker.update(_frame(), frame_index=2, timestamp_ms=33.0)

    assert update.tracked is True
    assert update.failed is False
    moved = update.detections[0]
    assert (moved.x1, moved.y1, moved.x2, moved.y2) == (
        13.0,
        18.0,
        33.0,
        48.0,
    )


def test_tracker_reports_failure_when_flow_points_are_lost() -> None:
    tracker = OpticalFlowTracker(
        TrackerConfig(), cv2_module=_FakeCv2(fail=True)
    )
    tracker.initialize(_frame(), (_person(),), frame_index=1, timestamp_ms=0.0)

    update = tracker.update(_frame(), frame_index=2, timestamp_ms=33.0)

    assert update.failed is True
    assert update.detections == ()
    assert tracker.stats()["failures"] == 1
