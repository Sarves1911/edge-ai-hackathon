import json
from pathlib import Path

from edge_detector.parity import bbox_iou, compare_jsonl


def _write_jsonl(path: Path, detections: list[dict]) -> None:
    path.write_text(
        json.dumps({"frame_index": 0, "detections": detections}) + "\n",
        encoding="utf-8",
    )


def _detection(*, confidence: float = 0.8, box: list[float] | None = None) -> dict:
    return {
        "class_id": 0,
        "class_name": "person",
        "confidence": confidence,
        "bbox_xyxy": [10.0, 10.0, 20.0, 20.0] if box is None else box,
    }


def test_identical_boxes_have_unit_iou() -> None:
    assert bbox_iou([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) == 1.0


def test_compare_passes_small_backend_differences(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_jsonl(reference, [_detection()])
    _write_jsonl(
        candidate,
        [_detection(confidence=0.79, box=[10.1, 10.1, 20.1, 20.1])],
    )

    summary = compare_jsonl(reference, candidate)
    assert summary["passed"] is True
    assert summary["matched_detections"] == 1


def test_compare_fails_missing_detection(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_jsonl(reference, [_detection()])
    _write_jsonl(candidate, [])

    summary = compare_jsonl(reference, candidate)
    assert summary["passed"] is False
    assert summary["unmatched_reference"] == 1

