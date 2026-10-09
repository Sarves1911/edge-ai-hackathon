# Partner handoff · Oct 8, 2026

## Completed checkpoint

- Stock YOLOv8n with person-only output at 256 x 320.
- PyTorch and static ONNX parity gate on fixed inputs.
- ONNX Runtime CPU benchmark and reproducible JSONL/config/summary artifacts.
- Motion signal with startup suppression and cooldown.
- Adaptive scheduler with startup, motion-event, active-cadence, hold, and idle-poll
  policies.
- Explicit cached-detection age and inference-reason records.
- One-slot latest-frame live-camera buffer with stale-frame replacement.
- Capture-to-result latency, replacement, duty-cycle, motion-score, cold-start, and
  steady-state metrics.
- Unit/integration suite: **32 passing tests**.

See `VALIDATION.md` for the frozen parity, adaptive-replay, and live-backpressure
measurements.

## Recording ownership

Record separate clips for these scenarios:

1. Empty static scene.
2. Person enters, stops for several seconds, and exits.
3. Person moves continuously across the frame.
4. Sudden lighting change with no person movement.
5. Background movement or minor camera shake.

Keep calibration clips separate from final evaluation clips. Do not tune motion
thresholds or INT8 calibration on the final evaluation set.

## Run every clip

Always-on control:

```bash
uv run --extra export edge-detect run \
  --source recordings/<clip>.mp4 \
  --config config/adaptive.yaml \
  --mode always \
  --output runs/eval-always \
  --no-annotated
```

Adaptive candidate:

```bash
uv run --extra export edge-detect run \
  --source recordings/<clip>.mp4 \
  --config config/adaptive.yaml \
  --mode adaptive \
  --output runs/eval-adaptive
```

Recorded files remain sequential even though `config/adaptive.yaml` selects the
latest-frame policy for live cameras.

## Motion-validation gate

Before changing thresholds, compare `motion_gate.score_p95` and `score_max` across all
five scenarios. Then freeze one configuration that satisfies all of the following:

- no repeated events in the empty static scene;
- an entry produces `motion_event` and prompt fresh inference;
- continued movement produces `active_interval` calls;
- a stopped person remains represented during the hold period;
- exit motion triggers a fresh result that clears the person;
- lighting changes and camera shake do not create sustained false activity.

Do not compare always-on and adaptive JSONL files with the backend parity command.
Adaptive mode intentionally skips inference and reuses explicitly marked cached results.

## After motion validation

1. Use only the calibration split with `tools/model/quantize_onnx.py`.
2. Compare FP32 and INT8 on the untouched evaluation split.
3. Keep INT8 only if it satisfies the model-size, latency, recall, and failure-case gates
   in `tools/model/README.md`.
4. Run the same FP32/INT8 and sequential/latest comparisons on the UNO Q when it arrives.
