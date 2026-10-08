from __future__ import annotations

from time import perf_counter
from typing import Any, Protocol

from .config import ModelConfig
from .contracts import Detection


class Detector(Protocol):
    def detect(self, frame: Any) -> tuple[tuple[Detection, ...], float]: ...


class YoloDetector:
    """Ultralytics reference backend.

    It deliberately owns preprocessing, inference, decoding, and NMS. The later ONNX/C++
    implementation must match this backend's boxes on a fixed golden input set.
    """

    def __init__(self, config: ModelConfig):
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is not installed. Run `uv sync` from the project root."
            ) from error

        self._config = config
        self._model = YOLO(config.path)

    def detect(self, frame: Any) -> tuple[tuple[Detection, ...], float]:
        predict_args: dict[str, Any] = {
            "source": frame,
            "imgsz": list(self._config.imgsz),
            "conf": self._config.confidence,
            "iou": self._config.iou,
            "classes": (
                None if self._config.classes is None else list(self._config.classes)
            ),
            # Keep native PyTorch and static exported backends on the exact same
            # height/width canvas. Ultralytics' rectangular auto-padding would otherwise
            # give the PyTorch reference a different tensor for portrait/landscape inputs.
            "rect": False,
            "verbose": False,
            "save": False,
        }
        if self._config.device is not None:
            predict_args["device"] = self._config.device

        started = perf_counter()
        predictions = self._model.predict(**predict_args)
        detector_ms = (perf_counter() - started) * 1_000.0

        if not predictions:
            return (), detector_ms
        result = predictions[0]
        if result.boxes is None:
            return (), detector_ms

        names = result.names
        xyxy = result.boxes.xyxy.detach().cpu().tolist()
        scores = result.boxes.conf.detach().cpu().tolist()
        class_ids = result.boxes.cls.detach().cpu().tolist()

        detections = []
        for box, score, raw_class_id in zip(xyxy, scores, class_ids, strict=True):
            class_id = int(raw_class_id)
            if isinstance(names, dict):
                class_name = str(names.get(class_id, class_id))
            else:
                class_name = str(names[class_id])
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(score),
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                )
            )
        return tuple(detections), detector_ms
