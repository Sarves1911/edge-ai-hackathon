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
    total_detections: int = 0

    def add(self, *, detector_ms: float, frame_ms: float, detections: int) -> None:
        self.detector_ms.append(detector_ms)
        self.frame_ms.append(frame_ms)
        self.total_detections += detections

    def summary(self, elapsed_s: float | None = None) -> dict[str, Any]:
        elapsed = perf_counter() - self.started_at if elapsed_s is None else elapsed_s
        frames = len(self.frame_ms)
        return {
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

