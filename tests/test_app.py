import json
from pathlib import Path

import numpy as np

import edge_detector.app as app_module
from edge_detector.config import (
    AppConfig,
    MotionConfig,
    OutputConfig,
    PipelineConfig,
    SchedulerConfig,
)
from edge_detector.contracts import Detection
from edge_detector.media import FramePacket


class _FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: np.ndarray) -> tuple[tuple[Detection, ...], float]:
        self.calls += 1
        return (), 5.0


class _FakeSource:
    fps = 10.0

    def __init__(self, kind: str = "video") -> None:
        self.kind = kind
        self.latest_only = False

    def __enter__(self) -> "_FakeSource":
        return self

    def frames(self, *, latest_only: bool = False):
        self.latest_only = latest_only
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        for index, timestamp_ms in enumerate((0.0, 100.0, 200.0, 1_000.0)):
            yield FramePacket(index=index, timestamp_ms=timestamp_ms, image=frame)

    def capture_summary(self) -> dict[str, float | int | str]:
        return {
            "mode": "latest" if self.latest_only else "sequential",
            "captured_frames": 4,
            "delivered_frames": 4,
            "replaced_frames": 0,
            "replacement_ratio": 0.0,
        }

    def __exit__(self, *_: object) -> None:
        return None


def test_adaptive_app_skips_detector_and_records_freshness(
    tmp_path: Path, monkeypatch
) -> None:
    detector = _FakeDetector()
    monkeypatch.setattr(app_module, "MediaSource", lambda _: _FakeSource())
    config = AppConfig(
        motion=MotionConfig(enabled=True),
        scheduler=SchedulerConfig(
            mode="adaptive",
            startup_frames=1,
            idle_poll_ms=1_000.0,
        ),
        output=OutputConfig(
            directory=str(tmp_path), save_annotated=False, save_jsonl=True
        ),
    )

    run_directory, summary = app_module.run_detection(
        config,
        "fake.mp4",
        detector_factory=lambda _: detector,
    )

    records = [
        json.loads(line)
        for line in (run_directory / "detections.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert detector.calls == 2
    assert summary["inference"]["calls"] == 2
    assert summary["inference"]["skipped_frames"] == 2
    assert [record["inference"]["reason"] for record in records] == [
        "startup",
        "idle_skip",
        "idle_skip",
        "idle_poll",
    ]
    assert records[1]["timing"]["detector_ms"] is None
    assert records[1]["inference"]["result_age_ms"] == 100.0
    assert summary["capture"]["mode"] == "sequential"


def test_app_enables_latest_buffer_only_for_camera(
    tmp_path: Path, monkeypatch
) -> None:
    detector = _FakeDetector()
    source = _FakeSource(kind="camera")
    injected_delays: list[float] = []
    monkeypatch.setattr(app_module, "MediaSource", lambda _: source)
    monkeypatch.setattr(app_module, "sleep", injected_delays.append)
    config = AppConfig(
        pipeline=PipelineConfig(
            live_capture_mode="latest", simulated_detector_delay_ms=100.0
        ),
        output=OutputConfig(
            directory=str(tmp_path), save_annotated=False, save_jsonl=False
        ),
    )

    _, summary = app_module.run_detection(
        config,
        "0",
        detector_factory=lambda _: detector,
    )

    assert source.latest_only is True
    assert summary["capture"]["mode"] == "latest"
    assert injected_delays == [0.1, 0.1, 0.1, 0.1]
