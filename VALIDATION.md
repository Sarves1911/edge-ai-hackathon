# Validation snapshot · Oct 6, 2026

The first vertical slice was validated in a Linux CPU-only workspace.

- Unit/contract tests: **11 passed**.
- Input: the Ultralytics `bus.jpg` package asset.
- Native reference: YOLOv8n PyTorch, static 256 x 320 preprocessing.
- Candidate: the same weights exported to ONNX, batch 1, static 256 x 320,
  opset 17, then executed with ONNX Runtime.
- Both backends returned the same six detections after the 0.25 confidence filter.
- All six class-matched boxes passed the parity gate.
- Minimum matched-box IoU: **0.9999983**.
- Mean matched-box IoU: **0.9999994**.
- Maximum confidence difference: **0.0000011**.

The one-frame latency from this run is intentionally not recorded as a benchmark: it was
a cold CPU invocation in a cloud workspace. Benchmark claims require warm-up plus a fixed
multi-frame clip on the actual laptop and UNO Q.

## Adaptive replay checkpoint · Oct 8, 2026

Environment: MacBook Pro CPU, ONNX Runtime 1.30.0 with CPUExecutionProvider,
`yolov8n.onnx`, person-only output, and the same 300-frame `webcam_raw.mp4` input.

| Metric | Always-on | Adaptive |
| --- | ---: | ---: |
| Processed frames | 300 | 300 |
| Detector calls | 300 | 12 |
| Inference duty cycle | 100% | 4% |
| Replay elapsed time | 2.378 s | 0.877 s |
| Replay throughput | 126.2 FPS | 342.1 FPS |
| Cached-result age p95 | 0 ms | 1,866.7 ms |

The adaptive scheduler reduced detector invocations by **96%** and replay time by
**63%** on this static clip. All adaptive calls came from three startup inferences and
nine idle safety polls. The clip generated no active-motion frames or motion events, so
this validates the idle/static path only; motion-triggered wake-up remains a required
test on the new recordings.

Replay throughput is pipeline processing throughput, not live camera FPS. Adaptive
detector percentiles contain only 12 samples and are distorted by the first lazy ONNX
invocation; use the separately reported first-inference and steady-state measurements.

## Live latest-frame checkpoint · Oct 8, 2026

Environment: MacBook Pro camera 0, always-on inference, CPU ONNX Runtime, 100 ms
injected detector delay, 50 delivered frames, and annotation disabled.

| Metric | Sequential capture | Latest-frame capture |
| --- | ---: | ---: |
| Captured frames | 50 | 204 |
| Delivered frames | 50 | 50 |
| Replaced stale frames | 0 | 152 |
| Replacement ratio | 0% | 74.5% |
| Elapsed time | 8.492 s | 7.349 s |
| Delivered throughput | 5.89 FPS | 6.80 FPS |
| Frame-wall p50 | 123.23 ms | 123.28 ms |
| Capture-to-result p95 | 127.68 ms | 158.15 ms |

The producer sustained approximately **27.8 captured FPS** while the consumer delivered
approximately **6.8 FPS**. The one-slot buffer replaced 152 unread frames instead of
allowing a stale backlog or unbounded memory growth. Two additional captured frames were
buffered or in flight when the bounded 50-frame run stopped.

Capture-to-result time begins when OpenCV returns a frame. It does not include any hidden
waiting inside a camera driver, so the sequential and latest latency values are not a
direct sensor-to-result comparison. The latest p95 is consistent with roughly 123 ms of
processing plus up to one camera-frame interval of buffer wait.

The live scene remained static. A large startup motion score was suppressed by the
three-frame warm-up; no active frames or motion events were emitted after startup.

