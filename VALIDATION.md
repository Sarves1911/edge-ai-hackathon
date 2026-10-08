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

