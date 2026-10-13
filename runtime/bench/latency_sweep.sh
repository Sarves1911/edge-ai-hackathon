#!/usr/bin/env bash
# Latency-injection sweep: the adaptive pipeline with the simulated detector replaying cached
# real detections (from an every-frame run) at 0/100/200/300/500 ms of injected latency.
# Replay is paced in real time so the detector is genuinely late relative to capture.
#
#   runtime/bench/latency_sweep.sh recording.mp4 [cached detections.jsonl]
#
# Without a cached file, an every-frame ONNX run (R2 config) is produced first.
set -euo pipefail

RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${RUNTIME_DIR}/.." && pwd)"
BIN="${RUNTIME_DIR}/build/app/edge_ai_runtime"

CLIP="$1"
name="$(basename "${CLIP%.*}")"
OUT="${REPO_DIR}/runs/latency/${name}"
CACHED="${2:-}"
mkdir -p "${OUT}"

if [[ -z "${CACHED}" ]]; then
    "${BIN}" --config "${RUNTIME_DIR}/config/ladder/r2.json" --source "replay:${CLIP}" \
        --output "${OUT}/reference" 2>&1 | grep -v swscaler
    CACHED="${OUT}/reference/detections.jsonl"
fi

for delay in 0 100 200 300 500; do
    echo "=== delay ${delay} ms"
    "${BIN}" --config "${RUNTIME_DIR}/config/runtime.json" --source "replay:${CLIP}" --realtime \
        --detector simulated --sim-jsonl "${CACHED}" --sim-delay-ms "${delay}" \
        --no-overlay --viewer-port 0 --output "${OUT}/delay-${delay}" 2>&1 | grep -v swscaler
done

python3 "${RUNTIME_DIR}/bench/summarize.py" "${OUT}" --reference "${CACHED}" | tee "${OUT}/scorecard.md"
