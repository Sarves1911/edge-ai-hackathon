from edge_detector.contracts import (
    Detection,
    FrameResult,
    InferenceStatus,
    MotionResult,
)


def test_frame_result_schema() -> None:
    detection = Detection(
        class_id=0,
        class_name="person",
        confidence=0.91,
        x1=10.0,
        y1=20.0,
        x2=30.0,
        y2=40.0,
    )
    result = FrameResult(
        frame_index=7,
        timestamp_ms=233.3,
        width=640,
        height=480,
        detector_ms=12.5,
        detections=(detection,),
    ).to_dict()

    assert result["frame_index"] == 7
    assert result["frame"] == {"width": 640, "height": 480}
    assert result["detections"][0]["bbox_xyxy"] == [10.0, 20.0, 30.0, 40.0]
    assert result["detections"][0]["class_name"] == "person"
    assert result["inference"]["ran"] is True
    assert result["inference"]["result_age_ms"] == 0.0



def test_frame_result_can_include_motion_signal() -> None:
    motion = MotionResult(
        score=0.125,
        changed_pixels=2400,
        active=True,
        event=True,
        processing_ms=0.4,
    )
    result = FrameResult(
        frame_index=1,
        timestamp_ms=100.0,
        width=640,
        height=480,
        detector_ms=5.0,
        detections=(),
        motion=motion,
    ).to_dict()

    assert result["motion"]["active"] is True
    assert result["motion"]["event"] is True
    assert result["motion"]["changed_pixels"] == 2400


def test_frame_result_marks_reused_detections_as_stale() -> None:
    result = FrameResult(
        frame_index=2,
        timestamp_ms=200.0,
        width=640,
        height=480,
        detector_ms=None,
        detections=(),
        inference=InferenceStatus(
            ran=False,
            reason="idle_skip",
            result_age_ms=100.0,
            reused_detections=True,
        ),
    ).to_dict()

    assert result["timing"]["detector_ms"] is None
    assert result["inference"] == {
        "ran": False,
        "reason": "idle_skip",
        "result_age_ms": 100.0,
        "reused_detections": True,
    }
