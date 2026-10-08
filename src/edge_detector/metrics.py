from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    if not 0.0 <= percent <= 100.0:
        raise ValueError("percent must be between 0 and 100")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@dataclass
class RunMetrics:
    model_load_ms: float
    started_at: float = field(default_factory=perf_counter)
    detector_ms: list[float] = field(default_factory=list)
    frame_ms: list[float] = field(default_factory=list)
    motion_ms: list[float] = field(default_factory=list)
    motion_scores: list[float] = field(default_factory=list)
    motion_active_frames: int = 0
    motion_events: int = 0
    total_detections: int = 0

    def add(
        self,
        *,
        detector_ms: float,
        frame_ms: float,
        detections: int,
        motion_ms: float | None = None,
        motion_score: float | None = None,
        motion_active: bool = False,
        motion_event: bool = False,
    ) -> None:
        self.detector_ms.append(detector_ms)
        self.frame_ms.append(frame_ms)
        self.total_detections += detections
        if motion_ms is not None:
            self.motion_ms.append(motion_ms)
        if motion_score is not None:
            self.motion_scores.append(motion_score)
        if motion_active:
            self.motion_active_frames += 1
        if motion_event:
            self.motion_events += 1

    def summary(self, elapsed_s: float | None = None) -> dict[str, Any]:
        elapsed = perf_counter() - self.started_at if elapsed_s is None else elapsed_s
        frames = len(self.frame_ms)
        summary: dict[str, Any] = {
            "frames": frames,
            "detections": self.total_detections,
            "elapsed_s": elapsed,
            "throughput_fps": 0.0 if elapsed <= 0 else frames / elapsed,
            "model_load_ms": self.model_load_ms,
            "detector_ms": {
                "p50": percentile(self.detector_ms, 50),
                "p95": percentile(self.detector_ms, 95),
                "mean": (
                    0.0
                    if not self.detector_ms
                    else sum(self.detector_ms) / len(self.detector_ms)
                ),
            },
            "frame_wall_ms": {
                "p50": percentile(self.frame_ms, 50),
                "p95": percentile(self.frame_ms, 95),
                "mean": (
                    0.0 if not self.frame_ms else sum(self.frame_ms) / len(self.frame_ms)
                ),
            },
        }
        if self.motion_ms:
            summary["motion_gate"] = {
                "active_frames": self.motion_active_frames,
                "events": self.motion_events,
                "active_frame_ratio": (
                    0.0 if frames == 0 else self.motion_active_frames / frames
                ),
                "score_mean": sum(self.motion_scores) / len(self.motion_scores),
                "processing_ms": {
                    "p50": percentile(self.motion_ms, 50),
                    "p95": percentile(self.motion_ms, 95),
                    "mean": sum(self.motion_ms) / len(self.motion_ms),
                },
            }
        return summary
