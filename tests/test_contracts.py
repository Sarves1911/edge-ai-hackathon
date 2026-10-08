from edge_detector.contracts import Detection, FrameResult


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

