from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox_xyxy": [self.x1, self.y1, self.x2, self.y2],
        }


@dataclass(frozen=True)
class MotionResult:
    score: float
    changed_pixels: int
    active: bool
    event: bool
    processing_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "changed_pixels": self.changed_pixels,
            "active": self.active,
            "event": self.event,
            "processing_ms": self.processing_ms,
        }


@dataclass(frozen=True)
class InferenceStatus:
    ran: bool = True
    reason: str = "always"
    result_age_ms: float | None = 0.0
    reused_detections: bool = False
    requested: bool | None = None
    request_accepted: bool | None = None
    result_frame_index: int | None = None
    result_timestamp_ms: float | None = None
    tracker_used: bool | None = None
    tracker_failed: bool | None = None
    stale_result_dropped: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "ran": self.ran,
            "reason": self.reason,
            "result_age_ms": self.result_age_ms,
            "reused_detections": self.reused_detections,
        }
        optional = {
            "requested": self.requested,
            "request_accepted": self.request_accepted,
            "result_frame_index": self.result_frame_index,
            "result_timestamp_ms": self.result_timestamp_ms,
            "tracker_used": self.tracker_used,
            "tracker_failed": self.tracker_failed,
            "stale_result_dropped": self.stale_result_dropped,
        }
        record.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        return record


@dataclass(frozen=True)
class FrameResult:
    frame_index: int
    timestamp_ms: float
    width: int
    height: int
    detector_ms: float | None
    detections: tuple[Detection, ...]
    motion: MotionResult | None = None
    inference: InferenceStatus = InferenceStatus()
    capture_to_result_ms: float | None = None
    detector_queue_ms: float | None = None
    tracker_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        record = {
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "frame": {"width": self.width, "height": self.height},
            "timing": {
                "detector_ms": self.detector_ms,
                "detector_queue_ms": self.detector_queue_ms,
                "tracker_ms": self.tracker_ms,
                "capture_to_result_ms": self.capture_to_result_ms,
            },
            "inference": self.inference.to_dict(),
            "detections": [detection.to_dict() for detection in self.detections],
        }
        if self.motion is not None:
            record["motion"] = self.motion.to_dict()
        return record

