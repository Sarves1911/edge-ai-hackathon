#!/usr/bin/env python3
"""Post-training static INT8 quantization for a fixed-shape detector ONNX model.

Calibration inputs use the same YOLO-style letterbox preprocessing as runtime:
BGR -> RGB, resize with aspect ratio, pad with 114, NCHW float32 in [0, 1].
"""

from __future__ import annotations

import argparse
import math
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate and quantize an FP32 detector ONNX model to INT8 QDQ."
    )
    parser.add_argument("--model", type=Path, required=True, help="FP32 ONNX model")
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        required=True,
        help="Calibration video, image, or image directory; repeat as needed",
    )
    parser.add_argument("--output", type=Path, required=True, help="INT8 ONNX output")
    parser.add_argument(
        "--samples",
        type=int,
        default=128,
        help="Maximum calibration frames across all sources (default: 128)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Override square input size; required only for a dynamic-shape model",
    )
    return parser.parse_args()


def model_input(model_path: Path, imgsz: int | None) -> tuple[str, int, int]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError(f"Expected one model input, found {len(inputs)}")

    model_input = inputs[0]
    if model_input.type != "tensor(float)":
        raise ValueError(f"Expected FP32 model input, found {model_input.type!r}")
    if len(model_input.shape) != 4:
        raise ValueError(f"Expected NCHW input, found shape {model_input.shape}")

    if imgsz is not None:
        return model_input.name, imgsz, imgsz

    height, width = model_input.shape[-2:]
    if not isinstance(height, int) or not isinstance(width, int):
        raise TypeError(
            f"Dynamic input shape {model_input.shape}; pass --imgsz explicitly"
        )
    return model_input.name, height, width


def letterbox(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    src_h, src_w = frame.shape[:2]
    scale = min(width / src_w, height / src_h)
    resized_w = max(1, round(src_w * scale))
    resized_h = max(1, round(src_h * scale))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w = width - resized_w
    pad_h = height - resized_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor[None, ...])


def image_paths(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    if source.suffix.lower() in IMAGE_SUFFIXES:
        return [source]
    return []


def evenly_spaced_indices(total: int, count: int) -> set[int]:
    count = min(total, count)
    if count <= 0:
        return set()
    return set(np.linspace(0, total - 1, count, dtype=np.int64).tolist())


def iter_source_frames(source: Path, quota: int) -> Iterator[np.ndarray]:
    images = image_paths(source)
    if images:
        for index in sorted(evenly_spaced_indices(len(images), quota)):
            frame = cv2.imread(str(images[index]), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Could not read image: {images[index]}")
            yield frame
        return

    if source.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported calibration source: {source}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        wanted = evenly_spaced_indices(total, quota) if total > 0 else set()
        frame_index = 0
        emitted = 0
        while emitted < quota:
            ok, frame = capture.read()
            if not ok:
                break
            take = frame_index in wanted if wanted else frame_index % 15 == 0
            if take:
                emitted += 1
                yield frame
            frame_index += 1
    finally:
        capture.release()


def iter_calibration_frames(
    sources: Sequence[Path], samples: int
) -> Iterator[np.ndarray]:
    quota = math.ceil(samples / len(sources))
    emitted = 0
    for source in sources:
        for frame in iter_source_frames(source, quota):
            yield frame
            emitted += 1
            if emitted >= samples:
                return


class DetectorCalibrationReader(CalibrationDataReader):
    def __init__(
        self,
        input_name: str,
        sources: Sequence[Path],
        samples: int,
        height: int,
        width: int,
    ) -> None:
        self._input_name = input_name
        self._sources = tuple(sources)
        self._samples = samples
        self._height = height
        self._width = width
        self._iterator: Iterator[dict[str, np.ndarray]] | None = None
        self.rewind()

    def _items(self) -> Iterator[dict[str, np.ndarray]]:
        count = 0
        for frame in iter_calibration_frames(self._sources, self._samples):
            count += 1
            yield {self._input_name: letterbox(frame, self._height, self._width)}
        if count == 0:
            raise RuntimeError("No calibration frames were read")

    def get_next(self) -> dict[str, np.ndarray] | None:
        assert self._iterator is not None
        return next(self._iterator, None)

    def rewind(self) -> None:
        self._iterator = self._items()


def size_mib(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    for source in args.source:
        if not source.exists():
            raise FileNotFoundError(source)

    input_name, height, width = model_input(args.model, args.imgsz)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reader = DetectorCalibrationReader(
        input_name=input_name,
        sources=args.source,
        samples=args.samples,
        height=height,
        width=width,
    )

    with tempfile.TemporaryDirectory(
        prefix=".quantize-", dir=args.output.parent
    ) as temporary_directory:
        preprocessed_model = Path(temporary_directory) / "preprocessed.onnx"
        quant_pre_process(
            input_model=str(args.model),
            output_model_path=str(preprocessed_model),
            skip_symbolic_shape=True,
        )
        quantize_static(
            model_input=str(preprocessed_model),
            model_output=str(args.output),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            per_channel=True,
            op_types_to_quantize=["Conv"],
        )

    onnx.checker.check_model(onnx.load(str(args.output)))
    ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    reduction = 100.0 * (1.0 - args.output.stat().st_size / args.model.stat().st_size)
    print(f"input={input_name} shape=1x3x{height}x{width}")
    print(f"fp32_mib={size_mib(args.model):.2f}")
    print(f"int8_mib={size_mib(args.output):.2f}")
    print(f"size_reduction_pct={reduction:.1f}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
