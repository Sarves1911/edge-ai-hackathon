from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def bbox_iou(first: list[float], second: list[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def load_jsonl(path: str | Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            frame_index = int(record["frame_index"])
            if frame_index in records:
                raise ValueError(f"Duplicate frame_index {frame_index} at line {line_number}")
            records[frame_index] = record
    return records


def compare_jsonl(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    min_iou: float = 0.90,
    score_tolerance: float = 0.02,
    min_confidence: float = 0.25,
) -> dict[str, Any]:
    reference = load_jsonl(reference_path)
    candidate = load_jsonl(candidate_path)
    frame_ids = sorted(set(reference) | set(candidate))
    matched_ious: list[float] = []
    score_deltas: list[float] = []
    reference_count = 0
    candidate_count = 0
    unmatched_reference = 0
    unmatched_candidate = 0
    missing_frames: list[int] = []

    for frame_id in frame_ids:
        if frame_id not in reference or frame_id not in candidate:
            missing_frames.append(frame_id)
        reference_detections = [
            detection
            for detection in reference.get(frame_id, {}).get("detections", [])
            if float(detection["confidence"]) >= min_confidence
        ]
        candidate_detections = [
            detection
            for detection in candidate.get(frame_id, {}).get("detections", [])
            if float(detection["confidence"]) >= min_confidence
        ]
        reference_count += len(reference_detections)
        candidate_count += len(candidate_detections)
        unused_candidates = set(range(len(candidate_detections)))

        for expected in sorted(
            reference_detections,
            key=lambda detection: float(detection["confidence"]),
            reverse=True,
        ):
            same_class = [
                index
                for index in unused_candidates
                if int(candidate_detections[index]["class_id"])
                == int(expected["class_id"])
            ]
            if not same_class:
                unmatched_reference += 1
                continue
            best_index = max(
                same_class,
                key=lambda index: bbox_iou(
                    expected["bbox_xyxy"], candidate_detections[index]["bbox_xyxy"]
                ),
            )
            actual = candidate_detections[best_index]
            overlap = bbox_iou(expected["bbox_xyxy"], actual["bbox_xyxy"])
            if overlap < min_iou:
                unmatched_reference += 1
                continue
            unused_candidates.remove(best_index)
            matched_ious.append(overlap)
            score_deltas.append(
                abs(float(expected["confidence"]) - float(actual["confidence"]))
            )
        unmatched_candidate += len(unused_candidates)

    max_score_delta = max(score_deltas, default=0.0)
    passed = (
        not missing_frames
        and unmatched_reference == 0
        and unmatched_candidate == 0
        and max_score_delta <= score_tolerance
    )
    return {
        "passed": passed,
        "frames_compared": len(frame_ids),
        "reference_detections": reference_count,
        "candidate_detections": candidate_count,
        "matched_detections": len(matched_ious),
        "unmatched_reference": unmatched_reference,
        "unmatched_candidate": unmatched_candidate,
        "missing_frame_ids": missing_frames,
        "iou": {
            "minimum": min(matched_ious, default=0.0),
            "mean": (
                0.0 if not matched_ious else sum(matched_ious) / len(matched_ious)
            ),
            "required_minimum": min_iou,
        },
        "confidence": {
            "maximum_delta": max_score_delta,
            "allowed_delta": score_tolerance,
            "filter_minimum": min_confidence,
        },
    }

