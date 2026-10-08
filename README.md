# Edge Object Detector

A problem-agnostic object-detection baseline for the edge-AI hackathon. Right now it
detects all COCO classes with stock YOLOv8n. The eventual problem is deliberately kept
out of the capture/inference/output core.

## What works in R0

- One command accepts a still image, video, stream URL, or webcam index.
- YOLOv8n runs at a configurable input shape, confidence, IoU, device, and class set.
- Every run saves its effective config, per-frame detections as JSONL, an annotated image
  or video, and a timing summary.
- Bounding boxes in JSONL are always in the original input frame's pixel coordinates.
- ONNX or ncnn export uses the same configured static input shape.

This is the Python correctness reference. ONNX/C++/INT8 must match its boxes on fixed
inputs before they count as optimizations.

Current checkpoint: the unit/contract suite passes, a real image completes end to end,
and the first static ONNX export matches all six native detections with minimum box IoU
`0.999998`. See `VALIDATION.md` for the exact scope and caveat.

## Set up

Install [`uv`](https://docs.astral.sh/uv/) if needed, then from this directory:

```bash
uv sync --dev
uv run pytest
```

The first real run downloads `yolov8n.pt` automatically. To avoid internet dependency at
the event, run once beforehand and keep the downloaded weight file locally.

## Run it

Video:

```bash
uv run edge-detect run --source path/to/clip.mp4
```

Mac webcam (camera 0, using Metal when supported):

```bash
uv run edge-detect run --source 0 --device mps
```

Quick CPU smoke test on only 20 frames:

```bash
uv run edge-detect run --source path/to/clip.mp4 --device cpu --max-frames 20
```

Use only selected COCO class IDs without changing code:

```bash
uv run edge-detect run --source path/to/clip.mp4 --classes 0,2,3,5,7
```

Each run creates a directory under `runs/detect/` containing:

```text
effective_config.yaml
detections.jsonl
annotated.jpg or annotated.mp4
summary.json
```

The stable JSONL record is the handoff to whatever application logic we choose later.
Do not make later presence, counting, safety, inspection, or alert logic parse rendered
pixels.

## Export the first edge artifact

```bash
uv sync --extra export
uv run edge-detect export --model yolov8n.pt --format onnx --imgsz 256 320
```

Keep the exported model in `models/` locally. Large model binaries are ignored by git.

Run the same golden input through the native and exported models, then gate them:

```bash
uv run edge-detect compare \
  --reference runs/reference/detections.jsonl \
  --candidate runs/onnx/detections.jsonl \
  --min-iou 0.90 \
  --score-tolerance 0.02
```

The command exits nonzero if a frame/detection is missing, a matched box misses the IoU
floor, or confidence changes beyond the allowed tolerance.

## What we are intentionally not building yet

- A tracker, line crossing, room occupancy, or trajectory prediction.
- Custom training/fine-tuning before the target objects and failure cases are known.
- An adaptive scheduler before fixed-rate inference is benchmarked.
- TensorRT, because the current target is the Qualcomm/ARM UNO Q rather than NVIDIA.

The next gate is simple: run this on one representative video, inspect the boxes, and
save its output as our golden reference. Then implement and validate the ONNX runtime
path.
