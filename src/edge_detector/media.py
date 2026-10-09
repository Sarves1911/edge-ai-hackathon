from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from typing import Any, Iterator, TextIO

from .contracts import Detection, FrameResult
from .latest import LatestValueBuffer

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is not installed. Run `uv sync` from the project root."
        ) from error
    return cv2


@dataclass(frozen=True)
class FramePacket:
    index: int
    timestamp_ms: float
    image: Any
    captured_at_monotonic_ms: float | None = None


class MediaSource:
    def __init__(self, source: str):
        self.source = source
        self.kind = "video"
        self.fps = 10.0
        self._image: Any | None = None
        self._capture: Any | None = None
        self._opened_at_monotonic_ms: float | None = None
        self._capture_mode = "sequential"
        self._captured_frames = 0
        self._delivered_frames = 0
        self._latest_buffer: LatestValueBuffer[FramePacket] | None = None
        self._latest_stop = Event()
        self._latest_thread: Thread | None = None

    def __enter__(self) -> "MediaSource":
        cv2 = _cv2()
        self._latest_stop.clear()
        source_path = Path(self.source)
        if source_path.suffix.lower() in IMAGE_SUFFIXES:
            self.kind = "image"
            self._image = cv2.imread(str(source_path))
            if self._image is None:
                raise RuntimeError(f"Could not read image: {self.source}")
            self.fps = 1.0
            return self

        capture_source: str | int = self.source
        if self.source.isdecimal() and not source_path.exists():
            capture_source = int(self.source)
            self.kind = "camera"
        self._capture = cv2.VideoCapture(capture_source)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open video/camera source: {self.source}")
        measured_fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self.fps = measured_fps if measured_fps > 0.0 else 10.0
        self._opened_at_monotonic_ms = perf_counter() * 1_000.0
        return self

    def _make_packet(self, index: int, frame: Any) -> FramePacket:
        captured_at_ms = perf_counter() * 1_000.0
        if self.kind == "camera":
            assert self._opened_at_monotonic_ms is not None
            timestamp_ms = captured_at_ms - self._opened_at_monotonic_ms
        else:
            cv2 = _cv2()
            assert self._capture is not None
            timestamp_ms = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
            if timestamp_ms <= 0.0 and index > 0:
                timestamp_ms = index * 1_000.0 / self.fps
        return FramePacket(
            index=index,
            timestamp_ms=timestamp_ms,
            image=frame,
            captured_at_monotonic_ms=captured_at_ms,
        )

    def _sequential_frames(self) -> Iterator[FramePacket]:
        self._capture_mode = "sequential"
        if self.kind == "image":
            self._captured_frames += 1
            self._delivered_frames += 1
            yield FramePacket(
                index=0,
                timestamp_ms=0.0,
                image=self._image,
                captured_at_monotonic_ms=perf_counter() * 1_000.0,
            )
            return

        assert self._capture is not None
        index = 0
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            packet = self._make_packet(index, frame)
            self._captured_frames += 1
            self._delivered_frames += 1
            yield packet
            index += 1

    def _capture_latest(self, buffer: LatestValueBuffer[FramePacket]) -> None:
        assert self._capture is not None
        index = 0
        try:
            while not self._latest_stop.is_set():
                ok, frame = self._capture.read()
                if not ok:
                    break
                packet = self._make_packet(index, frame)
                self._captured_frames += 1
                if not buffer.publish(packet):
                    break
                index += 1
        finally:
            buffer.close()

    def _latest_frames(self) -> Iterator[FramePacket]:
        if self.kind != "camera":
            yield from self._sequential_frames()
            return

        self._capture_mode = "latest"
        buffer: LatestValueBuffer[FramePacket] = LatestValueBuffer()
        self._latest_buffer = buffer
        thread = Thread(
            target=self._capture_latest,
            args=(buffer,),
            name="edge-camera-capture",
            daemon=True,
        )
        self._latest_thread = thread
        thread.start()
        try:
            while True:
                packet = buffer.take()
                if packet is None:
                    if buffer.closed:
                        break
                    continue
                self._delivered_frames += 1
                yield packet
        finally:
            self._latest_stop.set()
            buffer.close()

    def frames(self, *, latest_only: bool = False) -> Iterator[FramePacket]:
        if latest_only:
            yield from self._latest_frames()
        else:
            yield from self._sequential_frames()

    def capture_summary(self) -> dict[str, float | int | str]:
        replaced = (
            0
            if self._latest_buffer is None
            else self._latest_buffer.stats()["replaced"]
        )
        return {
            "mode": self._capture_mode,
            "captured_frames": self._captured_frames,
            "delivered_frames": self._delivered_frames,
            "replaced_frames": replaced,
            "replacement_ratio": (
                0.0
                if self._captured_frames == 0
                else replaced / self._captured_frames
            ),
        }

    def __exit__(self, *_: object) -> None:
        self._latest_stop.set()
        if self._latest_buffer is not None:
            self._latest_buffer.close()
        if self._latest_thread is not None:
            self._latest_thread.join(timeout=1.0)
            if self._latest_thread.is_alive() and self._capture is not None:
                # Some camera backends need release() to unblock a pending read.
                self._capture.release()
                self._latest_thread.join(timeout=1.0)
        if self._capture is not None:
            self._capture.release()


def draw_detections(
    frame: Any,
    detections: tuple[Detection, ...],
    thickness: int,
    *,
    cached_age_ms: float | None = None,
) -> Any:
    cv2 = _cv2()
    annotated = frame.copy()
    for detection in detections:
        start = (round(detection.x1), round(detection.y1))
        end = (round(detection.x2), round(detection.y2))
        color = (37, 211, 102) if cached_age_ms is None else (0, 165, 255)
        cv2.rectangle(annotated, start, end, color, thickness)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        if cached_age_ms is not None:
            label += f" cached {cached_age_ms:.0f}ms"
        text_y = max(18, start[1] - 7)
        cv2.putText(
            annotated,
            label,
            (start[0], text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            max(1, thickness - 1),
            cv2.LINE_AA,
        )
    return annotated


class RunSink:
    def __init__(
        self,
        *,
        run_directory: Path,
        source_kind: str,
        source_fps: float,
        save_annotated: bool,
        save_jsonl: bool,
        line_thickness: int,
    ):
        self.run_directory = run_directory
        self.source_kind = source_kind
        self.source_fps = source_fps
        self.save_annotated = save_annotated
        self.line_thickness = line_thickness
        self._writer: Any | None = None
        self._jsonl: TextIO | None = None
        self.annotated_path: Path | None = None
        self.jsonl_path: Path | None = None
        if save_jsonl:
            self.jsonl_path = run_directory / "detections.jsonl"
            self._jsonl = self.jsonl_path.open("w", encoding="utf-8")

    def write(self, packet: FramePacket, result: FrameResult) -> None:
        if self._jsonl is not None:
            self._jsonl.write(json.dumps(result.to_dict(), separators=(",", ":")) + "\n")

        if not self.save_annotated:
            return
        cv2 = _cv2()
        cached_age_ms = (
            result.inference.result_age_ms
            if result.inference.reused_detections
            else None
        )
        annotated = draw_detections(
            packet.image,
            result.detections,
            self.line_thickness,
            cached_age_ms=cached_age_ms,
        )
        if self.source_kind == "image":
            self.annotated_path = self.run_directory / "annotated.jpg"
            if not cv2.imwrite(str(self.annotated_path), annotated):
                raise RuntimeError(f"Could not write {self.annotated_path}")
            return

        if self._writer is None:
            height, width = annotated.shape[:2]
            self.annotated_path = self.run_directory / "annotated.mp4"
            codec = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self.annotated_path), codec, self.source_fps, (width, height)
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"Could not create {self.annotated_path}")
        self._writer.write(annotated)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
        if self._jsonl is not None:
            self._jsonl.close()

