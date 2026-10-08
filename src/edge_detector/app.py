from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from .backend import YoloDetector
from .config import AppConfig, config_to_dict
from .contracts import FrameResult
from .media import MediaSource, RunSink
from .metrics import RunMetrics
from .motion import MotionGate


def _source_stem(source: str) -> str:
    if source.isdecimal():
        return f"camera-{source}"
    stem = Path(source).stem or "stream"
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip("-.")
    return cleaned or "source"


def create_run_directory(base_directory: str, source: str) -> Path:
    base = Path(base_directory)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    candidate = base / f"{_source_stem(source)}-{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = base / f"{_source_stem(source)}-{timestamp}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def run_detection(config: AppConfig, source_name: str) -> tuple[Path, dict[str, Any]]:
    run_directory = create_run_directory(config.output.directory, source_name)
    (run_directory / "effective_config.yaml").write_text(
        yaml.safe_dump(config_to_dict(config), sort_keys=False), encoding="utf-8"
    )

    model_load_started = perf_counter()
    detector = YoloDetector(config.model)
    model_load_ms = (perf_counter() - model_load_started) * 1_000.0
    metrics = RunMetrics(model_load_ms=model_load_ms)
    motion_gate = MotionGate(config.motion) if config.motion.enabled else None

    sink: RunSink | None = None
    run_started = perf_counter()
    try:
        with MediaSource(source_name) as source:
            sink = RunSink(
                run_directory=run_directory,
                source_kind=source.kind,
                source_fps=source.fps,
                save_annotated=config.output.save_annotated,
                save_jsonl=config.output.save_jsonl,
                line_thickness=config.output.line_thickness,
            )
            for packet in source.frames():
                frame_started = perf_counter()
                motion = (
                    None
                    if motion_gate is None
                    else motion_gate.update(packet.image, packet.timestamp_ms)
                )
                detections, detector_ms = detector.detect(packet.image)
                height, width = packet.image.shape[:2]
                result = FrameResult(
                    frame_index=packet.index,
                    timestamp_ms=packet.timestamp_ms,
                    width=int(width),
                    height=int(height),
                    detector_ms=detector_ms,
                    detections=detections,
                    motion=motion,
                )
                sink.write(packet, result)
                frame_ms = (perf_counter() - frame_started) * 1_000.0
                metrics.add(
                    detector_ms=detector_ms,
                    frame_ms=frame_ms,
                    detections=len(detections),
                    motion_ms=None if motion is None else motion.processing_ms,
                    motion_score=None if motion is None else motion.score,
                    motion_active=False if motion is None else motion.active,
                    motion_event=False if motion is None else motion.event,
                )
                if (
                    config.pipeline.max_frames is not None
                    and len(metrics.frame_ms) >= config.pipeline.max_frames
                ):
                    break
    finally:
        if sink is not None:
            sink.close()

    if not metrics.frame_ms:
        raise RuntimeError("The source produced no frames")
    summary = metrics.summary(elapsed_s=perf_counter() - run_started)
    summary["source"] = source_name
    summary["run_directory"] = str(run_directory)
    summary["annotated_output"] = (
        None if sink is None or sink.annotated_path is None else str(sink.annotated_path)
    )
    summary["detections_jsonl"] = (
        None if sink is None or sink.jsonl_path is None else str(sink.jsonl_path)
    )
    (run_directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return run_directory, summary

