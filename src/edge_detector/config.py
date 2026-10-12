from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    path: str = "yolov8n.pt"
    imgsz: tuple[int, int] = (256, 320)
    confidence: float = 0.25
    iou: float = 0.45
    classes: tuple[int, ...] | None = None
    device: str | None = None


@dataclass(frozen=True)
class MotionConfig:
    enabled: bool = False
    width: int = 160
    height: int = 120
    pixel_threshold: int = 25
    minimum_changed_area: int = 192
    cooldown_ms: float = 250.0


@dataclass(frozen=True)
class SchedulerConfig:
    mode: str = "always"
    active_interval_ms: float = 100.0
    active_hold_ms: float = 1_500.0
    idle_poll_ms: float = 2_000.0
    startup_frames: int = 3


@dataclass(frozen=True)
class TrackerConfig:
    enabled: bool = True
    max_corners_per_box: int = 40
    min_points: int = 4
    quality_level: float = 0.01
    min_distance: float = 5.0
    block_size: int = 7
    window_size: int = 21
    max_level: int = 3
    criteria_iterations: int = 20
    criteria_epsilon: float = 0.03
    max_point_residual: float = 12.0
    max_result_age_ms: float = 1_000.0
    max_result_frame_gap: int = 30


@dataclass(frozen=True)
class PipelineConfig:
    live_capture_mode: str = "sequential"
    simulated_detector_delay_ms: float = 0.0
    max_frames: int | None = None


@dataclass(frozen=True)
class OutputConfig:
    directory: str = "runs/detect"
    save_annotated: bool = True
    save_jsonl: bool = True
    line_thickness: int = 2


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig = ModelConfig()
    motion: MotionConfig = MotionConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    tracker: TrackerConfig = TrackerConfig()
    pipeline: PipelineConfig = PipelineConfig()
    output: OutputConfig = OutputConfig()


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown key(s) in {section}: {names}")


def _parse_imgsz(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        size = (value, value)
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        size = (int(value[0]), int(value[1]))
    else:
        raise ValueError("model.imgsz must be one integer or [height, width]")
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError("model.imgsz dimensions must be positive")
    return size


def _parse_classes(value: Any) -> tuple[int, ...] | None:
    if value is None or value == "all":
        return None
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("model.classes must be null/'all' or a non-empty list")
    classes = tuple(int(class_id) for class_id in value)
    if any(class_id < 0 for class_id in classes):
        raise ValueError("model.classes cannot contain negative IDs")
    return classes


def _validate(config: AppConfig) -> AppConfig:
    if not 0.0 <= config.model.confidence <= 1.0:
        raise ValueError("model.confidence must be between 0 and 1")
    if not 0.0 <= config.model.iou <= 1.0:
        raise ValueError("model.iou must be between 0 and 1")
    if config.motion.width <= 0 or config.motion.height <= 0:
        raise ValueError("motion width/height must be positive")
    if not 1 <= config.motion.pixel_threshold <= 255:
        raise ValueError("motion.pixel_threshold must be between 1 and 255")
    total_motion_pixels = config.motion.width * config.motion.height
    if not 1 <= config.motion.minimum_changed_area <= total_motion_pixels:
        raise ValueError(
            "motion.minimum_changed_area must be between 1 and motion.width * motion.height"
        )
    if config.motion.cooldown_ms < 0:
        raise ValueError("motion.cooldown_ms cannot be negative")
    if config.scheduler.mode not in {"always", "adaptive"}:
        raise ValueError("scheduler.mode must be 'always' or 'adaptive'")
    if config.scheduler.active_interval_ms <= 0:
        raise ValueError("scheduler.active_interval_ms must be positive")
    if config.scheduler.active_hold_ms < 0:
        raise ValueError("scheduler.active_hold_ms cannot be negative")
    if config.scheduler.idle_poll_ms <= 0:
        raise ValueError("scheduler.idle_poll_ms must be positive")
    if config.scheduler.startup_frames < 1:
        raise ValueError("scheduler.startup_frames must be at least 1")
    if config.scheduler.mode == "adaptive" and not config.motion.enabled:
        raise ValueError("adaptive scheduler mode requires motion.enabled=true")
    if config.tracker.max_corners_per_box < 1:
        raise ValueError("tracker.max_corners_per_box must be positive")
    if not 1 <= config.tracker.min_points <= config.tracker.max_corners_per_box:
        raise ValueError(
            "tracker.min_points must be between 1 and max_corners_per_box"
        )
    if not 0.0 < config.tracker.quality_level <= 1.0:
        raise ValueError("tracker.quality_level must be between 0 and 1")
    if config.tracker.min_distance < 0:
        raise ValueError("tracker.min_distance cannot be negative")
    if config.tracker.block_size < 2:
        raise ValueError("tracker.block_size must be at least 2")
    if config.tracker.window_size < 2:
        raise ValueError("tracker.window_size must be at least 2")
    if config.tracker.max_level < 0:
        raise ValueError("tracker.max_level cannot be negative")
    if config.tracker.criteria_iterations < 1:
        raise ValueError("tracker.criteria_iterations must be positive")
    if config.tracker.criteria_epsilon <= 0:
        raise ValueError("tracker.criteria_epsilon must be positive")
    if config.tracker.max_point_residual <= 0:
        raise ValueError("tracker.max_point_residual must be positive")
    if config.tracker.max_result_age_ms < 0:
        raise ValueError("tracker.max_result_age_ms cannot be negative")
    if config.tracker.max_result_frame_gap < 0:
        raise ValueError("tracker.max_result_frame_gap cannot be negative")
    if config.pipeline.live_capture_mode not in {"sequential", "latest"}:
        raise ValueError(
            "pipeline.live_capture_mode must be 'sequential' or 'latest'"
        )
    if config.pipeline.simulated_detector_delay_ms < 0:
        raise ValueError(
            "pipeline.simulated_detector_delay_ms cannot be negative"
        )
    if config.pipeline.max_frames is not None and config.pipeline.max_frames <= 0:
        raise ValueError("pipeline.max_frames must be positive or null")
    if config.output.line_thickness <= 0:
        raise ValueError("output.line_thickness must be positive")
    return config


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("The config root must be a mapping")
    _reject_unknown(
        raw,
        {"model", "motion", "scheduler", "tracker", "pipeline", "output"},
        "config root",
    )

    model_raw = raw.get("model", {}) or {}
    motion_raw = raw.get("motion", {}) or {}
    scheduler_raw = raw.get("scheduler", {}) or {}
    tracker_raw = raw.get("tracker", {}) or {}
    pipeline_raw = raw.get("pipeline", {}) or {}
    output_raw = raw.get("output", {}) or {}
    for section_name, section in (
        ("model", model_raw),
        ("motion", motion_raw),
        ("scheduler", scheduler_raw),
        ("tracker", tracker_raw),
        ("pipeline", pipeline_raw),
        ("output", output_raw),
    ):
        if not isinstance(section, dict):
            raise ValueError(f"{section_name} must be a mapping")

    _reject_unknown(
        model_raw,
        {"path", "imgsz", "confidence", "iou", "classes", "device"},
        "model",
    )
    _reject_unknown(
        motion_raw,
        {
            "enabled",
            "width",
            "height",
            "pixel_threshold",
            "minimum_changed_area",
            "cooldown_ms",
        },
        "motion",
    )
    _reject_unknown(
        scheduler_raw,
        {
            "mode",
            "active_interval_ms",
            "active_hold_ms",
            "idle_poll_ms",
            "startup_frames",
        },
        "scheduler",
    )
    _reject_unknown(
        tracker_raw,
        {
            "enabled",
            "max_corners_per_box",
            "min_points",
            "quality_level",
            "min_distance",
            "block_size",
            "window_size",
            "max_level",
            "criteria_iterations",
            "criteria_epsilon",
            "max_point_residual",
            "max_result_age_ms",
            "max_result_frame_gap",
        },
        "tracker",
    )
    _reject_unknown(
        pipeline_raw,
        {"live_capture_mode", "simulated_detector_delay_ms", "max_frames"},
        "pipeline",
    )
    _reject_unknown(
        output_raw,
        {"directory", "save_annotated", "save_jsonl", "line_thickness"},
        "output",
    )

    model = ModelConfig(
        path=str(model_raw.get("path", ModelConfig.path)),
        imgsz=_parse_imgsz(model_raw.get("imgsz", ModelConfig.imgsz)),
        confidence=float(model_raw.get("confidence", ModelConfig.confidence)),
        iou=float(model_raw.get("iou", ModelConfig.iou)),
        classes=_parse_classes(model_raw.get("classes", ModelConfig.classes)),
        device=(
            None
            if model_raw.get("device", ModelConfig.device) is None
            else str(model_raw["device"])
        ),
    )
    motion = MotionConfig(
        enabled=bool(motion_raw.get("enabled", MotionConfig.enabled)),
        width=int(motion_raw.get("width", MotionConfig.width)),
        height=int(motion_raw.get("height", MotionConfig.height)),
        pixel_threshold=int(
            motion_raw.get("pixel_threshold", MotionConfig.pixel_threshold)
        ),
        minimum_changed_area=int(
            motion_raw.get(
                "minimum_changed_area", MotionConfig.minimum_changed_area
            )
        ),
        cooldown_ms=float(motion_raw.get("cooldown_ms", MotionConfig.cooldown_ms)),
    )
    scheduler = SchedulerConfig(
        mode=str(scheduler_raw.get("mode", SchedulerConfig.mode)),
        active_interval_ms=float(
            scheduler_raw.get(
                "active_interval_ms", SchedulerConfig.active_interval_ms
            )
        ),
        active_hold_ms=float(
            scheduler_raw.get("active_hold_ms", SchedulerConfig.active_hold_ms)
        ),
        idle_poll_ms=float(
            scheduler_raw.get("idle_poll_ms", SchedulerConfig.idle_poll_ms)
        ),
        startup_frames=int(
            scheduler_raw.get("startup_frames", SchedulerConfig.startup_frames)
        ),
    )
    tracker = TrackerConfig(
        enabled=bool(tracker_raw.get("enabled", TrackerConfig.enabled)),
        max_corners_per_box=int(
            tracker_raw.get(
                "max_corners_per_box", TrackerConfig.max_corners_per_box
            )
        ),
        min_points=int(tracker_raw.get("min_points", TrackerConfig.min_points)),
        quality_level=float(
            tracker_raw.get("quality_level", TrackerConfig.quality_level)
        ),
        min_distance=float(
            tracker_raw.get("min_distance", TrackerConfig.min_distance)
        ),
        block_size=int(tracker_raw.get("block_size", TrackerConfig.block_size)),
        window_size=int(
            tracker_raw.get("window_size", TrackerConfig.window_size)
        ),
        max_level=int(tracker_raw.get("max_level", TrackerConfig.max_level)),
        criteria_iterations=int(
            tracker_raw.get(
                "criteria_iterations", TrackerConfig.criteria_iterations
            )
        ),
        criteria_epsilon=float(
            tracker_raw.get("criteria_epsilon", TrackerConfig.criteria_epsilon)
        ),
        max_point_residual=float(
            tracker_raw.get(
                "max_point_residual", TrackerConfig.max_point_residual
            )
        ),
        max_result_age_ms=float(
            tracker_raw.get(
                "max_result_age_ms", TrackerConfig.max_result_age_ms
            )
        ),
        max_result_frame_gap=int(
            tracker_raw.get(
                "max_result_frame_gap", TrackerConfig.max_result_frame_gap
            )
        ),
    )
    pipeline = PipelineConfig(
        live_capture_mode=str(
            pipeline_raw.get(
                "live_capture_mode", PipelineConfig.live_capture_mode
            )
        ),
        simulated_detector_delay_ms=float(
            pipeline_raw.get(
                "simulated_detector_delay_ms",
                PipelineConfig.simulated_detector_delay_ms,
            )
        ),
        max_frames=(
            None
            if pipeline_raw.get("max_frames", PipelineConfig.max_frames) is None
            else int(pipeline_raw["max_frames"])
        )
    )
    output = OutputConfig(
        directory=str(output_raw.get("directory", OutputConfig.directory)),
        save_annotated=bool(
            output_raw.get("save_annotated", OutputConfig.save_annotated)
        ),
        save_jsonl=bool(output_raw.get("save_jsonl", OutputConfig.save_jsonl)),
        line_thickness=int(
            output_raw.get("line_thickness", OutputConfig.line_thickness)
        ),
    )
    return _validate(
        AppConfig(
            model=model,
            motion=motion,
            scheduler=scheduler,
            tracker=tracker,
            pipeline=pipeline,
            output=output,
        )
    )


def apply_overrides(
    config: AppConfig,
    *,
    model_path: str | None = None,
    imgsz: tuple[int, int] | None = None,
    confidence: float | None = None,
    iou: float | None = None,
    classes: tuple[int, ...] | None | object = ...,
    device: str | None | object = ...,
    motion_enabled: bool | None = None,
    scheduler_mode: str | None = None,
    active_interval_ms: float | None = None,
    active_hold_ms: float | None = None,
    idle_poll_ms: float | None = None,
    startup_frames: int | None = None,
    live_capture_mode: str | None = None,
    simulated_detector_delay_ms: float | None = None,
    max_frames: int | None = None,
    output_directory: str | None = None,
    save_annotated: bool | None = None,
    save_jsonl: bool | None = None,
) -> AppConfig:
    model = replace(
        config.model,
        path=config.model.path if model_path is None else model_path,
        imgsz=config.model.imgsz if imgsz is None else imgsz,
        confidence=config.model.confidence if confidence is None else confidence,
        iou=config.model.iou if iou is None else iou,
        classes=config.model.classes if classes is ... else classes,
        device=config.model.device if device is ... else device,
    )
    motion = replace(
        config.motion,
        enabled=(
            config.motion.enabled if motion_enabled is None else motion_enabled
        ),
    )
    scheduler = replace(
        config.scheduler,
        mode=(
            config.scheduler.mode if scheduler_mode is None else scheduler_mode
        ),
        active_interval_ms=(
            config.scheduler.active_interval_ms
            if active_interval_ms is None
            else active_interval_ms
        ),
        active_hold_ms=(
            config.scheduler.active_hold_ms
            if active_hold_ms is None
            else active_hold_ms
        ),
        idle_poll_ms=(
            config.scheduler.idle_poll_ms
            if idle_poll_ms is None
            else idle_poll_ms
        ),
        startup_frames=(
            config.scheduler.startup_frames
            if startup_frames is None
            else startup_frames
        ),
    )
    pipeline = replace(
        config.pipeline,
        live_capture_mode=(
            config.pipeline.live_capture_mode
            if live_capture_mode is None
            else live_capture_mode
        ),
        simulated_detector_delay_ms=(
            config.pipeline.simulated_detector_delay_ms
            if simulated_detector_delay_ms is None
            else simulated_detector_delay_ms
        ),
        max_frames=config.pipeline.max_frames if max_frames is None else max_frames,
    )
    output = replace(
        config.output,
        directory=(
            config.output.directory
            if output_directory is None
            else output_directory
        ),
        save_annotated=(
            config.output.save_annotated
            if save_annotated is None
            else save_annotated
        ),
        save_jsonl=config.output.save_jsonl if save_jsonl is None else save_jsonl,
    )
    return _validate(
        AppConfig(
            model=model,
            motion=motion,
            scheduler=scheduler,
            tracker=config.tracker,
            pipeline=pipeline,
            output=output,
        )
    )


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    return {
        "model": {
            "path": config.model.path,
            "imgsz": list(config.model.imgsz),
            "confidence": config.model.confidence,
            "iou": config.model.iou,
            "classes": None if config.model.classes is None else list(config.model.classes),
            "device": config.model.device,
        },
        "motion": {
            "enabled": config.motion.enabled,
            "width": config.motion.width,
            "height": config.motion.height,
            "pixel_threshold": config.motion.pixel_threshold,
            "minimum_changed_area": config.motion.minimum_changed_area,
            "cooldown_ms": config.motion.cooldown_ms,
        },
        "scheduler": {
            "mode": config.scheduler.mode,
            "active_interval_ms": config.scheduler.active_interval_ms,
            "active_hold_ms": config.scheduler.active_hold_ms,
            "idle_poll_ms": config.scheduler.idle_poll_ms,
            "startup_frames": config.scheduler.startup_frames,
        },
        "tracker": {
            "enabled": config.tracker.enabled,
            "max_corners_per_box": config.tracker.max_corners_per_box,
            "min_points": config.tracker.min_points,
            "quality_level": config.tracker.quality_level,
            "min_distance": config.tracker.min_distance,
            "block_size": config.tracker.block_size,
            "window_size": config.tracker.window_size,
            "max_level": config.tracker.max_level,
            "criteria_iterations": config.tracker.criteria_iterations,
            "criteria_epsilon": config.tracker.criteria_epsilon,
            "max_point_residual": config.tracker.max_point_residual,
            "max_result_age_ms": config.tracker.max_result_age_ms,
            "max_result_frame_gap": config.tracker.max_result_frame_gap,
        },
        "pipeline": {
            "live_capture_mode": config.pipeline.live_capture_mode,
            "simulated_detector_delay_ms": (
                config.pipeline.simulated_detector_delay_ms
            ),
            "max_frames": config.pipeline.max_frames,
        },
        "output": {
            "directory": config.output.directory,
            "save_annotated": config.output.save_annotated,
            "save_jsonl": config.output.save_jsonl,
            "line_thickness": config.output.line_thickness,
        },
    }

