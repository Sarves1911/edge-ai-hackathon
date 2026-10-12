#!/usr/bin/env python3
"""Builds the benchmark scorecard from run directories (stdlib only).

    python3 runtime/bench/summarize.py runs/ladder [--reference path/to/detections.jsonl]

Every directory containing summary.json is one row. C++ runtime summaries and the Python
reference summary (R0) are both understood.

Person-detection quality is reported as agreement with an every-frame reference detector
(IoU >= 0.5 per frame): recall = reference persons still reported, precision = reported
boxes the reference agrees with. This is a regression measure, not ground-truth accuracy.
By default the reference is the r2 run (C++ every-frame, deployed model) next to each row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> dict[int, list[list[float]]]:
    frames: dict[int, list[list[float]]] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            frames[int(record["frame_index"])] = [
                d["bbox_xyxy"] for d in record.get("detections", []) if int(d.get("class_id", 0)) == 0
            ]
    return frames


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def agreement(candidate: dict, reference: dict, threshold: float = 0.5) -> tuple[float, float] | None:
    tp = fp = fn = 0
    for frame, ref_boxes in reference.items():
        cand_boxes = list(candidate.get(frame, []))
        for ref in ref_boxes:
            best = max(range(len(cand_boxes)), key=lambda i: iou(ref, cand_boxes[i]), default=None)
            if best is not None and iou(ref, cand_boxes[best]) >= threshold:
                tp += 1
                cand_boxes.pop(best)
            else:
                fn += 1
        fp += len(cand_boxes)
    if tp + fn == 0 and tp + fp == 0:
        return None
    recall = tp / (tp + fn) if tp + fn else 1.0
    precision = tp / (tp + fp) if tp + fp else 1.0
    return recall, precision


def row_from_summary(path: Path) -> dict:
    s = json.loads(path.read_text())
    run_dir = path.parent

    if "total_frames" in s:   # C++ runtime
        frames = s["total_frames"]
        calls = s["total_detector_calls"]
        return {
            "frames": frames,
            "calls": calls,
            "avoided": s["detector_calls_avoided_pct"],
            "calls_min": s["detector_calls_per_minute"],
            "avg_ms": s["average_inference_ms"],
            "p95_ms": s["p95_inference_ms"],
            "loop_p95": s["frame_ms"]["p95"],
            "fps": s["throughput_fps"],
            "cpu": s["cpu_utilization_pct"],
            "ram": s["peak_ram_mb"],
            "model_mb": s["model"]["size_mb"],
            "jsonl": run_dir / "detections.jsonl",
        }

    # Python reference (edge-detect run)
    frames = s["frames"]
    calls = s["inference"]["calls"]
    rss_file = run_dir.parent / "peak_rss_mb.txt"
    duration_s = frames / 30.0
    return {
        "frames": frames,
        "calls": calls,
        "avoided": 100.0 * (frames - calls) / frames if frames else 0.0,
        "calls_min": calls / (duration_s / 60.0) if duration_s else 0.0,
        "avg_ms": s["detector_ms"]["steady_state"]["mean"],
        "p95_ms": s["detector_ms"]["steady_state"]["p95"],
        "loop_p95": s["frame_wall_ms"]["p95"],
        "fps": s["throughput_fps"],
        "cpu": None,
        "ram": float(rss_file.read_text()) if rss_file.exists() and rss_file.read_text().strip() else None,
        "model_mb": None,
        "jsonl": Path(s["detections_jsonl"]) if Path(s["detections_jsonl"]).is_absolute() else run_dir / "detections.jsonl",
    }


def fmt(value, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value}{suffix}"
    return f"{value:.{digits}f}{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--reference", type=Path, help="every-frame detections.jsonl for quality agreement")
    args = parser.parse_args()

    summaries = sorted(args.root.rglob("summary.json"))
    if not summaries:
        print(f"no summary.json under {args.root}", file=sys.stderr)
        return 1

    header = (
        "| Run | Frames | Detector calls | Calls avoided | Calls/min | Avg inf ms | P95 inf ms "
        "| Loop p95 ms | FPS | CPU % | Peak RAM MB | Model MB | Recall vs ref | Precision vs ref |"
    )
    print(header)
    print("|" + "|".join(["---"] * (header.count("|") - 1)) + "|")

    reference_cache: dict[Path, dict] = {}
    for path in summaries:
        row = row_from_summary(path)
        label = str(path.parent.relative_to(args.root))

        reference_path = args.reference
        if reference_path is None:
            # <root>/<clip>/r2/detections.jsonl next to this run
            parts = path.relative_to(args.root).parts
            if len(parts) >= 2:
                candidate = args.root / parts[0] / "r2" / "detections.jsonl"
                reference_path = candidate if candidate.exists() else None

        quality = None
        if reference_path is not None and row["jsonl"].exists():
            if reference_path not in reference_cache:
                reference_cache[reference_path] = load_jsonl(reference_path)
            quality = agreement(load_jsonl(row["jsonl"]), reference_cache[reference_path])

        print(
            f"| {label} | {row['frames']} | {row['calls']} | {fmt(row['avoided'], 1, '%')} | {fmt(row['calls_min'])} "
            f"| {fmt(row['avg_ms'], 2)} | {fmt(row['p95_ms'], 2)} | {fmt(row['loop_p95'], 2)} | {fmt(row['fps'])} "
            f"| {fmt(row['cpu'], 0)} | {fmt(row['ram'])} | {fmt(row['model_mb'])} "
            f"| {fmt(None if quality is None else 100 * quality[0], 1, '%')} "
            f"| {fmt(None if quality is None else 100 * quality[1], 1, '%')} |"
        )

    print("\nAvg/P95 power and energy: fill in from the board power meter (same duration per rung).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
