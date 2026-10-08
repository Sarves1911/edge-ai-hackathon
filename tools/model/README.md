# Model optimization slice

This keeps the existing YOLOv8n ONNX detector and adds one controlled experiment:
post-training static INT8 quantization. It does not train a new detector or change the
runtime architecture.

The network is still the 80-class COCO YOLOv8n model. The runtime retains only class `0`
(`person`). Describe this as **person-only output**, not as a one-class network or a
compute reduction; the full detection head still runs.

## 1. Verify the existing export dependencies

The project already has an `export` optional-dependency group and OpenCV in the runtime.
From the repository root:

```bash
uv sync --extra export
uv run --extra export python tools/model/quantize_onnx.py --help
```

Only if that reports a missing `onnx` or `onnxruntime` import, add the missing packages
to the existing group with `uv add --optional export ...`; do not install both
`opencv-python` and `opencv-python-headless` in the same environment.

## 2. Build the INT8 candidate

Start with the existing 300-frame recording. Add more `--source` arguments when the five
evaluation scenarios have been recorded.

```bash
uv run --extra export python tools/model/quantize_onnx.py \
  --model yolov8n.onnx \
  --source recordings/webcam_raw.mp4 \
  --samples 128 \
  --output models/yolov8n-int8.onnx
```

Do not commit generated model weights. Record the model name, SHA-256, export settings,
and measured size in the benchmark results instead.

## 3. Benchmark with the existing runner

```bash
uv run --extra export edge-detect run \
  --source recordings/webcam_raw.mp4 \
  --model models/yolov8n-int8.onnx \
  --device cpu \
  --output runs/bench-int8 \
  --no-annotated
```

## 4. Compare against the frozen FP32 reference

Replace the INT8 result directory timestamp below with the one printed by the runner.

```bash
uv run --extra export edge-detect compare \
  --reference runs/bench-pytorch/webcam_raw-20261006-214042/detections.jsonl \
  --candidate runs/bench-int8/<run>/detections.jsonl \
  --min-iou 0.85 \
  --score-tolerance 0.08
```

The comparison above is a regression check, not ground-truth accuracy. Before the final
model decision, evaluate FP32 and INT8 on the same labeled person-only set.

## Ship gate

Keep INT8 only when all of these hold on the UNO Q:

- model size falls by at least 50%;
- detector p95 latency improves by at least 10%;
- person recall drops by no more than 2 percentage points;
- no systematic missed-person or badly shifted-box failure appears in replay.

Otherwise ship FP32 and keep the INT8 measurements as an honest optimization result.
