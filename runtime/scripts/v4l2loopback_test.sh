#!/usr/bin/env bash
# Validates the real V4L2 capture path without a camera: a recording is streamed into a
# v4l2loopback device and read back through V4L2Source (poll + mmap + DQBUF/QBUF).
# Linux host only (needs the v4l2loopback kernel module and root).
#
#   runtime/scripts/v4l2loopback_test.sh [video] [model]
set -euo pipefail

RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${RUNTIME_DIR}/.." && pwd)"
VIDEO="${1:-${RUNTIME_DIR}/tests/data/test.mov}"
MODEL="${2:-${REPO_DIR}/models/yolov8n.onnx}"
DEVICE_NR=42
DEVICE="/dev/video${DEVICE_NR}"
BIN="${RUNTIME_DIR}/build/app/edge_ai_runtime"

[[ "$(uname -s)" == "Linux" ]] || { echo "v4l2loopback requires Linux" >&2; exit 1; }

sudo modprobe v4l2loopback devices=1 video_nr=${DEVICE_NR} card_label=edge_ai_loopback exclusive_caps=1

ffmpeg -loglevel error -re -stream_loop -1 -i "${VIDEO}" \
    -vf scale=640:480 -pix_fmt yuyv422 -f v4l2 "${DEVICE}" &
FEEDER=$!
trap 'kill ${FEEDER} 2>/dev/null || true' EXIT
sleep 2

EDGE_AI_V4L2_DEVICE="${DEVICE}" "${RUNTIME_DIR}/build/tests/v4l2_source_test"

DETECTOR_ARGS=(--detector simulated)
[[ -f "${MODEL}" ]] && DETECTOR_ARGS=(--model "${MODEL}")

"${BIN}" --config "${RUNTIME_DIR}/config/runtime.json" \
    --source "v4l2:${DEVICE}" "${DETECTOR_ARGS[@]}" \
    --max-frames 300 --output "${REPO_DIR}/runs/v4l2loopback"

grep -E '"camera_(captured|dropped|effective_fps)"' "${REPO_DIR}/runs/v4l2loopback/summary.json"
