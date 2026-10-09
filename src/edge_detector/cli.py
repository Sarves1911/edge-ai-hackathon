from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .app import run_detection
from .config import apply_overrides, load_config
from .parity import compare_jsonl


def _parse_classes(value: str) -> tuple[int, ...] | None:
    if value.strip().lower() == "all":
        return None
    try:
        classes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("classes must be 'all' or comma-separated IDs") from error
    if not classes or any(class_id < 0 for class_id in classes):
        raise argparse.ArgumentTypeError("classes must be 'all' or non-negative IDs")
    return classes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edge-detect",
        description="Problem-agnostic object detection for image, video, or webcam input.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the detector and write reproducible artifacts")
    run.add_argument("--source", required=True, help="Image, video, stream URL, or camera index")
    run.add_argument("--config", default="config/default.yaml")
    run.add_argument("--model")
    run.add_argument("--imgsz", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"))
    run.add_argument("--conf", type=float)
    run.add_argument("--iou", type=float)
    # SUPPRESS lets an explicit `--classes all` (parsed as None) remain distinct from
    # not providing the flag, so it can clear a class filter from the YAML config.
    run.add_argument("--classes", type=_parse_classes, default=argparse.SUPPRESS)
    run.add_argument("--device", help="Examples: cpu, mps, 0")
    run.add_argument(
        "--mode",
        choices=("always", "adaptive"),
        help="Always run inference or use motion-gated adaptive scheduling",
    )
    run.add_argument("--active-interval-ms", type=float)
    run.add_argument("--active-hold-ms", type=float)
    run.add_argument("--idle-poll-ms", type=float)
    run.add_argument("--startup-frames", type=int)
    run.add_argument(
        "--live-capture-mode",
        choices=("sequential", "latest"),
        help="Live-camera buffering policy; recorded files stay sequential",
    )
    run.add_argument(
        "--simulate-detector-delay-ms",
        type=float,
        help="Testing only: sleep before each detector call",
    )
    run.add_argument("--max-frames", type=int)
    run.add_argument("--output")
    run.add_argument("--no-annotated", action="store_true")
    run.add_argument("--no-jsonl", action="store_true")

    export = subparsers.add_parser("export", help="Export the reference model for edge runtimes")
    export.add_argument("--model", default="yolov8n.pt")
    export.add_argument("--format", default="onnx", choices=("onnx", "ncnn"))
    export.add_argument("--imgsz", nargs=2, type=int, default=(256, 320), metavar=("HEIGHT", "WIDTH"))
    export.add_argument("--opset", type=int, default=17)

    compare = subparsers.add_parser(
        "compare", help="Gate an exported backend against reference detection JSONL"
    )
    compare.add_argument("--reference", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--min-iou", type=float, default=0.90)
    compare.add_argument("--score-tolerance", type=float, default=0.02)
    compare.add_argument("--min-confidence", type=float, default=0.25)
    return parser


def _run_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    classes_override: tuple[int, ...] | None | object = ...
    if hasattr(args, "classes"):
        classes_override = args.classes
    config = apply_overrides(
        config,
        model_path=args.model,
        imgsz=None if args.imgsz is None else tuple(args.imgsz),
        confidence=args.conf,
        iou=args.iou,
        classes=classes_override,
        device=config.model.device if args.device is None else args.device,
        motion_enabled=True if args.mode == "adaptive" else None,
        scheduler_mode=args.mode,
        active_interval_ms=args.active_interval_ms,
        active_hold_ms=args.active_hold_ms,
        idle_poll_ms=args.idle_poll_ms,
        startup_frames=args.startup_frames,
        live_capture_mode=args.live_capture_mode,
        simulated_detector_delay_ms=args.simulate_detector_delay_ms,
        max_frames=args.max_frames,
        output_directory=args.output,
        save_annotated=False if args.no_annotated else None,
        save_jsonl=False if args.no_jsonl else None,
    )
    _, summary = run_detection(config, args.source)
    print(json.dumps(summary, indent=2))
    return 0


def _export_command(args: argparse.Namespace) -> int:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is not installed. Run `uv sync` from the project root."
        ) from error
    export_args = {
        "format": args.format,
        "imgsz": list(args.imgsz),
        "batch": 1,
        "dynamic": False,
    }
    if args.format == "onnx":
        export_args["opset"] = args.opset
    exported_path = YOLO(args.model).export(**export_args)
    print(json.dumps({"exported_model": str(Path(exported_path))}, indent=2))
    return 0


def _compare_command(args: argparse.Namespace) -> int:
    summary = compare_jsonl(
        args.reference,
        args.candidate,
        min_iou=args.min_iou,
        score_tolerance=args.score_tolerance,
        min_confidence=args.min_confidence,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return _run_command(args)
    if args.command == "export":
        return _export_command(args)
    if args.command == "compare":
        return _compare_command(args)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
