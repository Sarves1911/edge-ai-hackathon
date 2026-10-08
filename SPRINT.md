# Generic object-detection sprint · Oct 6–13, 2026

The original occupancy plan is reduced to one reusable edge pipeline:

```text
image/video/webcam -> detector -> boxes in original-frame coordinates
                    -> JSONL + optional overlay + benchmark summary
```

No tracker, line crossing, occupancy logic, or domain fine-tuning is allowed before the
problem statement is chosen.

## Oct 6 · R0 reference

- Freeze the detection JSONL and configuration contracts.
- Run stock YOLOv8n on one image/video at 320 x 256.
- Save annotated output, boxes, model-load time, detector p50/p95, wall-frame p50/p95,
  and throughput.
- Export a static batch-1 ONNX model.

Exit: one real input produces reviewable boxes and reproducible artifacts.

## Oct 7 · parity and replay

- Select 20 fixed frames as the golden set.
- Add the deployment runtime (ONNX Runtime or ncnn after checking the UNO Q image).
- Validate retained classes/counts and match corresponding boxes at IoU >= 0.90.
- Add preprocessing, decode, confidence/class-filter, NMS, and empty-output tests.
- Add video/replay input and a five-minute no-drop smoke test.

Exit: the deployable backend agrees with R0 and handles a complete clip.

## Oct 8 · INT8 and live execution

- Calibrate with roughly 300 representative frames.
- Keep INT8 only if detection accuracy falls by no more than two points on the fixed set.
- Add a live latest-frame queue: when inference is busy, replace stale work with the newest
  frame rather than building latency.
- Sweep 100/200/300/500 ms injected delay and report result age p50/p95 and effective FPS.

Exit: camera capture stays responsive even when inference is slower than the camera.

## Oct 9 · evaluation and feature freeze

- Record fixed evaluation clips and label boxes in COCO or YOLO format after the problem
  is chosen. Never tune on this set.
- Two-hour replay soak: bounded memory, no queue growth, monotonic timestamps.
- Test the board setup script in the ARM64 environment.
- Fill predicted benchmark rows and freeze features.

## Oct 10–11 · UNO Q

- Validate raw camera capture at 640 x 480, 10 FPS for five minutes.
- Benchmark runtime, FP32/INT8, input size, and 1/2/4 threads.
- Measure idle, camera, streaming, and inference power using the same meter boundary.
- Fill the ladder with latency p50/p95, throughput, peak RAM, model size, average watts,
  and joules per inference.
- Demo boxes, class, confidence, FPS, and latency; use replay as the fallback.

## Oct 12–13 · rehearse and ship

- Two complete offline rehearsals.
- Store models and dependencies locally.
- Record a backup demo video and keep replay mode ready.

## Results ladder

| Rung | One controlled change |
| --- | --- |
| R0 | Python + stock YOLOv8n FP32 at 640 |
| R1 | Same Python reference at 320 x 256 |
| R2 | Deployment runtime, FP32 at 320 x 256 |
| R3 | INT8 |
| R4 | Asynchronous latest-frame inference |
| R5 | Board thread/system tuning |
| R6 | One problem-specific event module after the problem is chosen |

Accuracy metrics become mAP50, mAP50-95, precision, and recall once a labeled target
set exists. Until then, report only runtime metrics; do not manufacture an accuracy claim.

