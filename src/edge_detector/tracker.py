from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from .config import TrackerConfig
from .contracts import Detection


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is not installed. Run `uv sync` from the project root."
        ) from error
    return cv2


@dataclass(frozen=True)
class TrackerUpdate:
    detections: tuple[Detection, ...]
    tracked: bool
    failed: bool
    processing_ms: float


@dataclass
class _TrackedBox:
    detection: Detection
    points: np.ndarray


class OpticalFlowTracker:
    """Sparse Lucas-Kanade propagation of detector-created boxes.

    The tracker never creates a semantic detection. YOLO initializes/corrects the
    boxes, and optical flow only translates those boxes between detector results.
    """

    def __init__(self, config: TrackerConfig, *, cv2_module: Any | None = None) -> None:
        self._config = config
        self._cv2 = _cv2() if cv2_module is None else cv2_module
        self._previous_gray: np.ndarray | None = None
        self._frame_index: int | None = None
        self._timestamp_ms: float | None = None
        self._tracks: list[_TrackedBox] = []
        self._initializations = 0
        self._updates = 0
        self._failures = 0

    @staticmethod
    def _valid_box(detection: Detection) -> bool:
        return detection.x2 > detection.x1 and detection.y2 > detection.y1

    def _gray(self, frame: Any) -> np.ndarray:
        if len(frame.shape) == 2:
            return frame
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)

    def _seed_points(
        self, gray: np.ndarray, detection: Detection
    ) -> np.ndarray:
        height, width = gray.shape[:2]
        x1 = max(0, min(width - 1, int(detection.x1)))
        y1 = max(0, min(height - 1, int(detection.y1)))
        x2 = max(x1 + 1, min(width, int(np.ceil(detection.x2))))
        y2 = max(y1 + 1, min(height, int(np.ceil(detection.y2))))
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return np.empty((0, 1, 2), dtype=np.float32)
        points = self._cv2.goodFeaturesToTrack(
            roi,
            maxCorners=self._config.max_corners_per_box,
            qualityLevel=self._config.quality_level,
            minDistance=self._config.min_distance,
            blockSize=self._config.block_size,
        )
        if points is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        points[:, 0, 0] += x1
        points[:, 0, 1] += y1
        return points

    def initialize(
        self,
        frame: Any,
        detections: tuple[Detection, ...],
        *,
        frame_index: int,
        timestamp_ms: float,
    ) -> TrackerUpdate:
        started = perf_counter()
        gray = self._gray(frame)
        self._tracks = [
            _TrackedBox(detection=detection, points=self._seed_points(gray, detection))
            for detection in detections
            if self._valid_box(detection)
        ]
        self._previous_gray = gray
        self._frame_index = frame_index
        self._timestamp_ms = timestamp_ms
        self._initializations += 1
        return TrackerUpdate(
            detections=self.detections,
            tracked=False,
            failed=False,
            processing_ms=(perf_counter() - started) * 1_000.0,
        )

    @staticmethod
    def _shift_detection(
        detection: Detection,
        dx: float,
        dy: float,
        *,
        width: int,
        height: int,
    ) -> Detection | None:
        box_width = detection.x2 - detection.x1
        box_height = detection.y2 - detection.y1
        x1 = min(max(0.0, detection.x1 + dx), max(0.0, width - box_width))
        y1 = min(max(0.0, detection.y1 + dy), max(0.0, height - box_height))
        x2 = min(float(width), x1 + box_width)
        y2 = min(float(height), y1 + box_height)
        if x2 <= x1 or y2 <= y1:
            return None
        return Detection(
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )

    def update(
        self,
        frame: Any,
        *,
        frame_index: int,
        timestamp_ms: float,
    ) -> TrackerUpdate:
        started = perf_counter()
        current_gray = self._gray(frame)

        if self._previous_gray is None:
            self._previous_gray = current_gray
            self._frame_index = frame_index
            self._timestamp_ms = timestamp_ms
            return TrackerUpdate((), False, False, (perf_counter() - started) * 1_000.0)

        if self._frame_index == frame_index:
            return TrackerUpdate(
                self.detections,
                False,
                False,
                (perf_counter() - started) * 1_000.0,
            )

        if not self._tracks:
            self._previous_gray = current_gray
            self._frame_index = frame_index
            self._timestamp_ms = timestamp_ms
            self._updates += 1
            return TrackerUpdate((), False, False, (perf_counter() - started) * 1_000.0)

        height, width = current_gray.shape[:2]
        surviving: list[_TrackedBox] = []
        failed = False
        for track in self._tracks:
            previous_points = track.points
            if len(previous_points) < self._config.min_points:
                previous_points = self._seed_points(
                    self._previous_gray, track.detection
                )
            if len(previous_points) < self._config.min_points:
                failed = True
                continue

            next_points, status, _ = self._cv2.calcOpticalFlowPyrLK(
                self._previous_gray,
                current_gray,
                previous_points,
                None,
                winSize=(self._config.window_size, self._config.window_size),
                maxLevel=self._config.max_level,
                criteria=(
                    self._cv2.TERM_CRITERIA_EPS | self._cv2.TERM_CRITERIA_COUNT,
                    self._config.criteria_iterations,
                    self._config.criteria_epsilon,
                ),
            )
            if next_points is None or status is None:
                failed = True
                continue
            valid = np.asarray(status).reshape(-1) == 1
            valid &= np.isfinite(np.asarray(next_points).reshape(-1, 2)).all(axis=1)
            old_valid = previous_points.reshape(-1, 2)[valid]
            new_valid = np.asarray(next_points, dtype=np.float32).reshape(-1, 2)[valid]
            if len(new_valid) < self._config.min_points:
                failed = True
                continue

            displacement = new_valid - old_valid
            median = np.median(displacement, axis=0)
            residual = np.linalg.norm(displacement - median, axis=1)
            inliers = residual <= self._config.max_point_residual
            if int(np.count_nonzero(inliers)) < self._config.min_points:
                failed = True
                continue
            good_points = new_valid[inliers]
            dx, dy = np.median(displacement[inliers], axis=0)
            shifted = self._shift_detection(
                track.detection,
                float(dx),
                float(dy),
                width=width,
                height=height,
            )
            if shifted is None:
                failed = True
                continue
            surviving.append(
                _TrackedBox(
                    detection=shifted,
                    points=good_points.reshape(-1, 1, 2).astype(np.float32),
                )
            )

        self._tracks = surviving
        self._previous_gray = current_gray
        self._frame_index = frame_index
        self._timestamp_ms = timestamp_ms
        self._updates += 1
        if failed:
            self._failures += 1
        return TrackerUpdate(
            detections=self.detections,
            tracked=bool(surviving),
            failed=failed,
            processing_ms=(perf_counter() - started) * 1_000.0,
        )

    @property
    def detections(self) -> tuple[Detection, ...]:
        return tuple(track.detection for track in self._tracks)

    @property
    def frame_index(self) -> int | None:
        return self._frame_index

    def reset(self) -> None:
        self._previous_gray = None
        self._frame_index = None
        self._timestamp_ms = None
        self._tracks = []

    def stats(self) -> dict[str, int]:
        return {
            "initializations": self._initializations,
            "updates": self._updates,
            "failures": self._failures,
            "active_boxes": len(self._tracks),
        }
