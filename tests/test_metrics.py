import pytest

from edge_detector.metrics import RunMetrics, percentile


def test_percentile_interpolates() -> None:
    assert percentile([10.0, 20.0, 30.0], 50) == 20.0
    assert percentile([10.0, 20.0], 95) == pytest.approx(19.5)


def test_run_summary() -> None:
    metrics = RunMetrics(model_load_ms=50.0)
    metrics.add(detector_ms=10.0, frame_ms=12.0, detections=2)
    metrics.add(detector_ms=20.0, frame_ms=24.0, detections=1)
    summary = metrics.summary(elapsed_s=0.5)

    assert summary["frames"] == 2
    assert summary["detections"] == 3
    assert summary["throughput_fps"] == 4.0
    assert summary["detector_ms"]["p50"] == 15.0



def test_run_summary_includes_motion_gate_metrics() -> None:
    metrics = RunMetrics(model_load_ms=10.0)
    metrics.add(
        detector_ms=5.0,
        frame_ms=6.0,
        detections=1,
        motion_ms=0.2,
        motion_score=0.10,
        motion_active=True,
        motion_event=True,
    )
    metrics.add(
        detector_ms=5.0,
        frame_ms=6.0,
        detections=1,
        motion_ms=0.1,
        motion_score=0.0,
    )
    summary = metrics.summary(elapsed_s=1.0)

    assert summary["motion_gate"]["active_frames"] == 1
    assert summary["motion_gate"]["events"] == 1
    assert summary["motion_gate"]["active_frame_ratio"] == 0.5
    assert summary["motion_gate"]["score_mean"] == 0.05
