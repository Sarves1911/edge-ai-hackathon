import pytest

from edge_detector.metrics import RunMetrics, percentile


def test_percentile_interpolates() -> None:
    assert percentile([10.0, 20.0, 30.0], 50) == 20.0
    assert percentile([10.0, 20.0], 95) == pytest.approx(19.5)


def test_run_summary() -> None:
    metrics = RunMetrics(model_load_ms=50.0)
    metrics.add(
        detector_ms=10.0,
        frame_ms=12.0,
        detections=2,
        capture_to_result_ms=14.0,
    )
    metrics.add(
        detector_ms=20.0,
        frame_ms=24.0,
        detections=1,
        capture_to_result_ms=26.0,
    )
    summary = metrics.summary(elapsed_s=0.5)

    assert summary["frames"] == 2
    assert summary["detections"] == 3
    assert summary["throughput_fps"] == 4.0
    assert summary["detector_ms"]["p50"] == 15.0
    assert summary["detector_ms"]["first"] == 10.0
    assert summary["detector_ms"]["steady_state"] == {
        "samples": 1,
        "p50": 20.0,
        "p95": 20.0,
        "mean": 20.0,
    }
    assert summary["inference"]["calls"] == 2
    assert summary["inference"]["duty_cycle"] == 1.0
    assert summary["capture_to_result_ms"]["p50"] == 20.0



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
    assert summary["motion_gate"]["score_p95"] == pytest.approx(0.095)
    assert summary["motion_gate"]["score_max"] == 0.10


def test_run_summary_tracks_skipped_and_reused_frames() -> None:
    metrics = RunMetrics(model_load_ms=10.0)
    metrics.add(
        detector_ms=5.0,
        frame_ms=6.0,
        detections=1,
        inference_ran=True,
        inference_reason="startup",
        result_age_ms=0.0,
    )
    metrics.add(
        detector_ms=None,
        frame_ms=0.2,
        detections=0,
        inference_ran=False,
        inference_reason="idle_skip",
        result_age_ms=100.0,
        reused_detections=True,
    )

    summary = metrics.summary(elapsed_s=1.0)

    assert summary["inference"]["calls"] == 1
    assert summary["inference"]["skipped_frames"] == 1
    assert summary["inference"]["duty_cycle"] == 0.5
    assert summary["inference"]["reused_detection_frames"] == 1
    assert summary["inference"]["reasons"] == {
        "idle_skip": 1,
        "startup": 1,
    }
    assert summary["inference"]["result_age_ms"]["p50"] == 50.0
