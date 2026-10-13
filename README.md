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

The adaptive path separates cheap motion measurement, sparse optical-flow tracking,
inference policy, and YOLO execution. A single detector worker accepts at most one
replaceable pending frame, so a slow detector cannot create a FIFO backlog. YOLO still
owns person identification; the tracker only moves YOLO-created boxes until the next
detector correction.

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

Adaptive recorded-video replay remains sequential and deterministic:

```bash
uv run --extra export edge-detect run \
  --source recordings/webcam_raw.mp4 \
  --config config/adaptive.yaml \
  --no-annotated
```

For a live camera, the same config activates both the one-slot latest-frame capture
buffer and asynchronous YOLO execution. Capture, motion, tracking, and scheduling
continue while the detector worker is busy. Both capture and detector overload replace
unread stale work with the newest frame instead of building latency:

```bash
uv run --extra export edge-detect run \
  --source 0 \
  --config config/adaptive.yaml
```

The summary reports captured/replaced frames, detector jobs submitted/replaced,
tracker failures/refreshes, and capture-to-result p50/p95 latency. Use
`--live-capture-mode sequential` for a controlled live-camera comparison. Recorded
replay also remains deterministic: inference runs on the worker thread, but replay waits
for each scheduled result.

To demonstrate overload behavior before the board arrives, inject detector delay while
running the live camera. This option is for measurement only and defaults to zero:

```bash
uv run --extra export edge-detect run \
  --source 0 \
  --config config/adaptive.yaml \
  --mode always \
  --simulate-detector-delay-ms 100 \
  --max-frames 100 \
  --no-annotated
```

Repeat with `100`, `200`, `300`, and `500` ms in both `sequential` and `latest`
capture modes. Compare replaced frames and capture-to-result p50/p95 rather than only
throughput.

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

The repository permits the three board-demo exports at `models/yolov8n.onnx`,
`models/yolov8n-int8.onnx`, and `models/yolov8n-640.onnx`; other model binaries remain
ignored. The standard demo recording path is `recordings/webcam_raw.mp4`.

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

- DeepSORT, ByteTrack, ReID, line crossing, room occupancy, or trajectory prediction.
- Historical frame replay to re-run tracking across every frame missed during inference.
- Custom training/fine-tuning before the target objects and failure cases are known.
- TensorRT, because the current target is the Qualcomm/ARM UNO Q rather than NVIDIA.

The next gate is the live-camera 100/200/300/500 ms delay sweep with the async worker and
tracker, followed by the same test on the target board.

## Board demo viewer

The C++ runtime includes a live browser view with server-rendered person boxes and pipeline
telemetry. The shipped runtime config enables it on port 8080:

```bash
runtime/build/app/edge_ai_runtime --config runtime/config/runtime.json \
  --source v4l2:/dev/video0 --model models/yolov8n.onnx
```

Open `http://<board-ip>:8080`, or tunnel it with
`ssh -L 8080:localhost:8080 <user>@<board-ip>` and open `http://localhost:8080`.
