# LeanVision C++ runtime

Camera-gated adaptive person detection for the Arduino UNO Q. The Python package in `src/`
remains the correctness reference; this runtime is the low-latency edge pipeline.

```text
FrameSource (replay | V4L2 poll/mmap)
    |
    +--> MotionGate (160x120 gray diff)
    +--> Tracker.predict (constant velocity, held still where nothing moved)
    +--> AdaptiveScheduler  -> IDLE | TRACK_ONLY | RUN_DETECTOR
              |
              | RUN_DETECTOR
              v
   preprocess into preallocated Tensor A/B  (frame memory released immediately)
              |
              v
   DetectorWorker thread -> YOLOv8 ONNX -> decode/NMS/unletterbox
              |
              v
   DetectionBatch(frame_ts_ns = source frame)
              |
              v
   Tracker.update: correct the historical state at frame_ts_ns, re-predict to now
```

## Build and test

```bash
brew install onnxruntime ffmpeg cmake        # macOS (Linux: see scripts/setup_board.sh)
cmake -S runtime -B runtime/build -DCMAKE_BUILD_TYPE=Release
cmake --build runtime/build -j
ctest --test-dir runtime/build --output-on-failure
```

Without ONNX Runtime the build still succeeds with the simulated detector only
(`-DEDGE_AI_WITH_ONNXRUNTIME=OFF`, or set `ONNXRUNTIME_ROOT` to an extracted release).

Export the model once through the Python reference:

```bash
mkdir -p models && cd models
uv run --project .. --extra export edge-detect export --model yolov8n.pt --format onnx --imgsz 256 320
```

For an offline board clone, the repository allowlist includes the standard artifacts
`models/yolov8n.onnx`, `models/yolov8n-int8.onnx`, `models/yolov8n-640.onnx`, and
`recordings/webcam_raw.mp4`. Keep each file below GitHub's normal per-file limit, or use
Git LFS instead of committing a partial upload.

## Run

```bash
# Adaptive pipeline on a recording, paced in real time, with the status line
runtime/build/app/edge_ai_runtime --config runtime/config/runtime.json \
    --source replay:recordings/clip.mp4 --realtime

# Live USB camera (Linux). runtime.json starts the annotated viewer on port 8080.
runtime/build/app/edge_ai_runtime --config runtime/config/runtime.json --source v4l2:/dev/video0

# Every-frame synchronous baseline
runtime/build/app/edge_ai_runtime --source replay:clip.mp4 --mode always --sync --output runs/cpp-always

# Latency injection with cached real detections
runtime/build/app/edge_ai_runtime --config runtime/config/runtime.json --source replay:clip.mp4 --realtime \
    --detector simulated --sim-jsonl runs/cpp-always/detections.jsonl --sim-delay-ms 300
```

For the live demo, open `http://<board-ip>:8080` on the Mac. The C++ runtime draws
tracked person rectangles into a compact JPEG feed and serves live state, action, reason,
inference time, detector-call count, and FPS beside it. The viewer has no authentication,
so use it only on a trusted demo network. If venue Wi-Fi blocks device-to-device traffic:

```bash
ssh -L 8080:localhost:8080 <board-user>@<board-ip>
# Then open http://localhost:8080
```

Override the defaults with `--viewer-port <port> --viewer-fps <1-30>`; port `0` disables
the viewer. `--no-overlay` only disables the terminal status line.

Each run directory contains `effective_config.json`, `frames.csv` (per-frame motion, action,
reason, state, timings), `detections.jsonl` (Python-compatible schema; tracked boxes carry
`track_id`) and `summary.json` (detector calls/avoided, inference p50/p95, result latency,
control-loop time, FPS, CPU %, peak RAM, model size/load time, inference reasons).

All thresholds live in `config/runtime.json`. Unknown keys and out-of-range values are rejected.

| Scheduler mode | Ladder rung | Behavior |
|---|---|---|
| `always` | R1–R3 | detector on every frame |
| `fixed_interval` | R4 | async detector + tracker, detector every `fixed_interval_ms` |
| `motion_gated` | R5 | fixed interval while the scene is active, idle polls when quiet |
| `adaptive` | R6 | motion, unexplained motion, tracker failure/weakness and semantic age decide |

## Parity with the Python reference

```bash
uv run --extra export edge-detect run --source clip.mp4 --model models/yolov8n.onnx \
    --device cpu --mode always --no-annotated --output runs/py-always
runtime/build/app/edge_ai_runtime --source replay:clip.mp4 --model models/yolov8n.onnx \
    --mode always --sync --no-overlay --output runs/cpp-always
uv run --extra export edge-detect compare --reference runs/py-always/<run>/detections.jsonl \
    --candidate runs/cpp-always/detections.jsonl --min-iou 0.90 --score-tolerance 0.02
```

## Benchmarks

```bash
runtime/bench/run_ladder.sh recordings/*.mp4        # R0-R6 + scorecard.md
runtime/bench/latency_sweep.sh recordings/clip.mp4  # 0/100/200/300/500 ms injected latency
```

`run_ladder.sh` exports the 640 and 256x320 models and the INT8 model
(`tools/model/quantize_onnx.py`) when missing. Use `--calibration` with a clip that is not in
the evaluation set, and `--realtime` when measuring CPU or power.

"Recall/precision vs ref" in the scorecard is agreement with the every-frame R2 detector at
IoU 0.5, not ground-truth accuracy.

## ARM64 and the UNO Q

```bash
runtime/scripts/build_arm64.sh                 # Debian trixie linux/arm64 container: build + ctest
runtime/scripts/setup_board.sh                 # on the board: deps, ONNX Runtime, build, tests, camera, model
runtime/scripts/v4l2loopback_test.sh           # Linux host: recording -> v4l2loopback -> V4L2Source
```

## Module map

| Module | Responsibility |
|---|---|
| `core` | `Frame`/`ImageView` (non-owning), monotonic time |
| `capture` | `FrameSource`, FFmpeg `ReplaySource`, Linux `V4L2Source` |
| `preprocess` | fused letterbox + bilinear + HWC→CHW + normalize (RGB24/BGR24/YUYV, stride-aware) |
| `detector` | `Detection`/`DetectionBatch`, `Detector`, YOLOv8 postprocess, `SimulatedDetector`, `DetectorWorker` |
| `onnx_detector` | ONNX Runtime backend (built only when ONNX Runtime is found) |
| `motion` | `MotionGate` score/active/event, unexplained motion, per-box change |
| `tracker` | tracks, association, lifecycle, timestamp-aware correction history |
| `scheduler` | `AdaptiveScheduler` policy and IDLE/RECHECK/ACTIVE state |
| `metrics` | `Profiler`: CSV/JSONL streaming, summary statistics, CPU/RAM |
| `config` | `RuntimeConfig` JSON loading and validation |
| `pipeline` | the capture/control loop composing the modules (used by the app and integration test) |
| `viewer` | annotated JPEG encoding plus the browser-based live demo server |

## Ownership rules

- A `Frame` returned by a source is valid only until the next `nextFrame()`. V4L2 buffers are
  requeued at the start of the next call. The pipeline copies what it needs (motion downsample,
  tensor) within the same step.
- Tensor A/B are allocated once in `DetectorWorker`. There is at most one request in flight and
  one pending. Under `latest` the pending request is replaced; under `skip` it is refused.
- `DetectionBatch` owns its detections and carries the source-frame timestamp.
