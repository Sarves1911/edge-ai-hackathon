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
class PipelineConfig:
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
    _reject_unknown(raw, {"model", "motion", "pipeline", "output"}, "config root")

    model_raw = raw.get("model", {}) or {}
    motion_raw = raw.get("motion", {}) or {}
    pipeline_raw = raw.get("pipeline", {}) or {}
    output_raw = raw.get("output", {}) or {}
    for section_name, section in (
        ("model", model_raw),
        ("motion", motion_raw),
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
    _reject_unknown(pipeline_raw, {"max_frames"}, "pipeline")
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
    pipeline = PipelineConfig(
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
        AppConfig(model=model, motion=motion, pipeline=pipeline, output=output)
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
    pipeline = replace(
        config.pipeline,
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
        AppConfig(model=model, motion=config.motion, pipeline=pipeline, output=output)
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
        "pipeline": {"max_frames": config.pipeline.max_frames},
        "output": {
            "directory": config.output.directory,
            "save_annotated": config.output.save_annotated,
            "save_jsonl": config.output.save_jsonl,
            "line_thickness": config.output.line_thickness,
        },
    }

