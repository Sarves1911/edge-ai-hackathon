from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import yaml

from .backend import Detector, YoloDetector
from .config import AppConfig, ModelConfig, config_to_dict
from .contracts import Detection, FrameResult, InferenceStatus
from .detector_worker import AsyncDetectorWorker, DetectorJob, DetectorResult
from .media import MediaSource, RunSink
from .metrics import RunMetrics
from .motion import MotionGate
from .scheduler import InferenceScheduler
from .tracker import OpticalFlowTracker, TrackerUpdate


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


def run_detection(
    config: AppConfig,
    source_name: str,
    *,
    detector_factory: Callable[[ModelConfig], Detector] = YoloDetector,
) -> tuple[Path, dict[str, Any]]:
    run_directory = create_run_directory(config.output.directory, source_name)
    (run_directory / "effective_config.yaml").write_text(
        yaml.safe_dump(config_to_dict(config), sort_keys=False), encoding="utf-8"
    )

    model_load_started = perf_counter()
    detector = detector_factory(config.model)
    model_load_ms = (perf_counter() - model_load_started) * 1_000.0
    metrics = RunMetrics(model_load_ms=model_load_ms)
    motion_gate = MotionGate(config.motion) if config.motion.enabled else None
    scheduler = InferenceScheduler(config.scheduler)
    tracker = (
        OpticalFlowTracker(config.tracker) if config.tracker.enabled else None
    )
    last_detections: tuple[Detection, ...] = ()
    last_inference_timestamp_ms: float | None = None
    last_result_frame_index: int | None = None
    stale_detector_results = 0
    tracker_refresh_requests = 0

    worker = AsyncDetectorWorker(
        detector,
        simulated_delay_ms=config.pipeline.simulated_detector_delay_ms,
        # Keeping this injectable preserves deterministic delay tests.
        sleep_fn=sleep,
    )
    worker.start()

    sink: RunSink | None = None
    capture_summary: dict[str, Any] | None = None
    worker_stopped_cleanly = False
    run_started = perf_counter()

    def apply_detector_result(
        detector_result: DetectorResult,
        packet: Any,
        *,
        enforce_freshness: bool,
        use_tracker: bool,
    ) -> tuple[bool, bool, bool, float]:
        """Apply a result and catch its boxes up to the current frame.

        Returns (applied, stale, tracker_failed, tracker_ms). Async results are
        initialized on their own owned frame, then propagated directly to the
        current frame. This avoids applying old coordinates as if they were fresh
        without adding a historical replay queue.
        """
        nonlocal last_detections
        nonlocal last_inference_timestamp_ms
        nonlocal last_result_frame_index

        if detector_result.error is not None:
            raise RuntimeError(
                f"Detector worker failed on frame {detector_result.frame_index}"
            ) from detector_result.error

        frame_gap = packet.index - detector_result.frame_index
        result_age_ms = packet.timestamp_ms - detector_result.timestamp_ms
        out_of_order = (
            last_result_frame_index is not None
            and detector_result.frame_index <= last_result_frame_index
        )
        stale = (
            out_of_order
            or frame_gap < 0
            or result_age_ms < 0.0
            or (
                enforce_freshness
                and (
                    result_age_ms > config.tracker.max_result_age_ms
                    or frame_gap > config.tracker.max_result_frame_gap
                )
            )
        )
        if stale:
            return False, True, False, 0.0

        tracker_failed = False
        tracker_ms = 0.0
        if tracker is None or not use_tracker:
            last_detections = detector_result.detections
        else:
            initialized = tracker.initialize(
                detector_result.image,
                detector_result.detections,
                frame_index=detector_result.frame_index,
                timestamp_ms=detector_result.timestamp_ms,
            )
            tracker_ms += initialized.processing_ms
            last_detections = initialized.detections
            if detector_result.frame_index != packet.index:
                caught_up = tracker.update(
                    packet.image,
                    frame_index=packet.index,
                    timestamp_ms=packet.timestamp_ms,
                )
                tracker_ms += caught_up.processing_ms
                tracker_failed = caught_up.failed
                last_detections = caught_up.detections

        last_inference_timestamp_ms = detector_result.timestamp_ms
        last_result_frame_index = detector_result.frame_index
        return True, False, tracker_failed, tracker_ms

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
            latest_only = (
                source.kind == "camera"
                and config.pipeline.live_capture_mode == "latest"
            )
            for packet in source.frames(latest_only=latest_only):
                frame_started = perf_counter()
                # Real live packets carry the monotonic capture clock populated by
                # MediaSource. Synthetic/replay-like packets remain deterministic.
                asynchronous_live = (
                    latest_only and packet.captured_at_monotonic_ms is not None
                )
                tracking_active = (
                    tracker is not None
                    and (
                        config.scheduler.mode == "adaptive"
                        or asynchronous_live
                    )
                )
                applied_result: DetectorResult | None = None
                tracker_used = False
                tracker_failed = False
                tracker_ms = 0.0
                stale_result_dropped = False

                ready_result = worker.take_result()
                if ready_result is not None:
                    applied, stale, failed, result_tracker_ms = apply_detector_result(
                        ready_result,
                        packet,
                        enforce_freshness=asynchronous_live,
                        use_tracker=tracking_active,
                    )
                    tracker_ms += result_tracker_ms
                    tracker_failed = failed
                    stale_result_dropped = stale
                    if applied:
                        applied_result = ready_result
                        tracker_used = ready_result.frame_index != packet.index
                    else:
                        stale_detector_results += 1

                if tracking_active and tracker.frame_index != packet.index:
                    tracked: TrackerUpdate = tracker.update(
                        packet.image,
                        frame_index=packet.index,
                        timestamp_ms=packet.timestamp_ms,
                    )
                    tracker_ms += tracked.processing_ms
                    tracker_used = tracker_used or tracked.tracked
                    tracker_failed = tracker_failed or tracked.failed
                    last_detections = tracked.detections

                motion = (
                    None
                    if motion_gate is None
                    else motion_gate.update(packet.image, packet.timestamp_ms)
                )
                decision = scheduler.decide(packet.timestamp_ms, motion)
                if tracker_failed:
                    decision = scheduler.force(packet.timestamp_ms, "tracker_failure")
                    tracker_refresh_requests += 1
                elif stale_result_dropped:
                    decision = scheduler.force(packet.timestamp_ms, "stale_result")

                request_accepted = False
                if decision.run_inference:
                    request_accepted = worker.submit(
                        DetectorJob.from_frame(
                            frame_index=packet.index,
                            timestamp_ms=packet.timestamp_ms,
                            image=packet.image,
                            reason=decision.reason,
                        )
                    )

                    # Recorded input and sequential-camera mode remain deterministic:
                    # inference executes on the worker thread, while this loop waits
                    # for that exact result. Latest-camera mode stays fully async.
                    if request_accepted and not asynchronous_live:
                        blocking_result = worker.take_result(wait=True)
                        if blocking_result is None:
                            raise RuntimeError("Detector worker stopped without a result")
                        applied, stale, failed, result_tracker_ms = apply_detector_result(
                            blocking_result,
                            packet,
                            enforce_freshness=False,
                            use_tracker=tracking_active,
                        )
                        tracker_ms += result_tracker_ms
                        tracker_failed = tracker_failed or failed
                        stale_result_dropped = stale_result_dropped or stale
                        if stale:
                            stale_detector_results += 1
                        if applied:
                            applied_result = blocking_result
                            tracker_used = False

                detections = last_detections
                detector_ms = (
                    None if applied_result is None else applied_result.detector_ms
                )
                detector_queue_ms = (
                    None if applied_result is None else applied_result.queue_wait_ms
                )

                result_age_ms = (
                    None
                    if last_inference_timestamp_ms is None
                    else max(0.0, packet.timestamp_ms - last_inference_timestamp_ms)
                )
                reused_detections = (
                    last_inference_timestamp_ms is not None
                    and (
                        applied_result is None
                        or applied_result.frame_index != packet.index
                        or tracker_used
                    )
                )
                capture_to_result_ms = (
                    None
                    if packet.captured_at_monotonic_ms is None
                    else max(
                        0.0,
                        perf_counter() * 1_000.0
                        - packet.captured_at_monotonic_ms,
                    )
                )
                height, width = packet.image.shape[:2]
                result = FrameResult(
                    frame_index=packet.index,
                    timestamp_ms=packet.timestamp_ms,
                    width=int(width),
                    height=int(height),
                    detector_ms=detector_ms,
                    detections=detections,
                    motion=motion,
                    inference=InferenceStatus(
                        ran=applied_result is not None,
                        reason=decision.reason,
                        result_age_ms=result_age_ms,
                        reused_detections=reused_detections,
                        requested=decision.run_inference,
                        request_accepted=request_accepted,
                        result_frame_index=last_result_frame_index,
                        result_timestamp_ms=last_inference_timestamp_ms,
                        tracker_used=tracker_used,
                        tracker_failed=tracker_failed,
                        stale_result_dropped=stale_result_dropped,
                    ),
                    capture_to_result_ms=capture_to_result_ms,
                    detector_queue_ms=detector_queue_ms,
                    tracker_ms=tracker_ms,
                )
                sink.write(packet, result)
                frame_ms = (perf_counter() - frame_started) * 1_000.0
                metrics.add(
                    detector_ms=detector_ms,
                    frame_ms=frame_ms,
                    detections=(
                        0
                        if applied_result is None
                        else len(applied_result.detections)
                    ),
                    motion_ms=None if motion is None else motion.processing_ms,
                    motion_score=None if motion is None else motion.score,
                    motion_active=False if motion is None else motion.active,
                    motion_event=False if motion is None else motion.event,
                    inference_ran=applied_result is not None,
                    inference_reason=decision.reason,
                    result_age_ms=result_age_ms,
                    reused_detections=reused_detections,
                    capture_to_result_ms=capture_to_result_ms,
                    tracker_ms=tracker_ms,
                )
                if (
                    config.pipeline.max_frames is not None
                    and len(metrics.frame_ms) >= config.pipeline.max_frames
                ):
                    break
        capture_summary = source.capture_summary()
    finally:
        worker_stopped_cleanly = worker.stop(drain=False)
        trailing_result = worker.take_result()
        if sink is not None:
            sink.close()
        if trailing_result is not None and trailing_result.error is not None:
            raise RuntimeError(
                f"Detector worker failed on frame {trailing_result.frame_index}"
            ) from trailing_result.error

    if not metrics.frame_ms:
        raise RuntimeError("The source produced no frames")
    detector_durations_ms, completed_detections = worker.detector_observations()
    metrics.finalize_detector_observations(
        detector_durations_ms, completed_detections
    )
    summary = metrics.summary(elapsed_s=perf_counter() - run_started)
    tracker_summary = summary.setdefault("tracker", {})
    if tracker is not None:
        tracker_summary.update(tracker.stats())
    tracker_summary.update(
        {
            "enabled": tracker is not None,
            "refresh_requests": tracker_refresh_requests,
            "stale_detector_results": stale_detector_results,
        }
    )
    summary["detector_worker"] = worker.stats()
    summary["detector_worker"]["stopped_cleanly"] = worker_stopped_cleanly
    summary["capture"] = capture_summary
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

