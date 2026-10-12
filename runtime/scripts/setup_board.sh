#!/usr/bin/env bash
# UNO Q bring-up: dependencies, ONNX Runtime, native build, unit tests, camera and model checks.
# Safe to re-run.
#
#   runtime/scripts/setup_board.sh [--ort-tarball path/to/onnxruntime-linux-aarch64-X.tgz]
#                                  [--model models/yolov8n.onnx] [--device /dev/video0]
set -euo pipefail

RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${RUNTIME_DIR}/.." && pwd)"
BUILD_DIR="${RUNTIME_DIR}/build"

ORT_VERSION="${ORT_VERSION:-1.31.0}"
ORT_ROOT="${ONNXRUNTIME_ROOT:-/opt/onnxruntime}"
ORT_TARBALL=""
MODEL="${REPO_DIR}/models/yolov8n.onnx"
DEVICE="/dev/video0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ort-tarball) ORT_TARBALL="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

step(){ printf '\n==> %s\n' "$*"; }

step "System"
uname -a
echo "arch: $(uname -m), cores: $(nproc)"
free -h || true

step "Dependencies"
SUDO=""
[[ $EUID -ne 0 ]] && SUDO="sudo"
${SUDO} apt-get update
${SUDO} apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config ca-certificates curl \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev v4l-utils

step "ONNX Runtime"
if [[ -f "${ORT_ROOT}/include/onnxruntime_cxx_api.h" ]]; then
    echo "found ${ORT_ROOT}"
else
    if [[ -z "${ORT_TARBALL}" ]]; then
        ORT_TARBALL="/tmp/onnxruntime-linux-aarch64-${ORT_VERSION}.tgz"
        [[ -f "${ORT_TARBALL}" ]] || curl -fL -o "${ORT_TARBALL}" \
            "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-aarch64-${ORT_VERSION}.tgz"
    fi
    TMP_DIR="$(mktemp -d)"
    tar -xzf "${ORT_TARBALL}" -C "${TMP_DIR}"
    ${SUDO} rm -rf "${ORT_ROOT}"
    ${SUDO} mv "${TMP_DIR}"/onnxruntime-* "${ORT_ROOT}"
    rmdir "${TMP_DIR}"
    echo "installed ${ORT_ROOT}"
fi
export ONNXRUNTIME_ROOT="${ORT_ROOT}"
export LD_LIBRARY_PATH="${ORT_ROOT}/lib:${LD_LIBRARY_PATH:-}"

step "Camera"
if command -v v4l2-ctl >/dev/null; then
    v4l2-ctl --list-devices || true
    if [[ -e "${DEVICE}" ]]; then
        v4l2-ctl -d "${DEVICE}" --list-formats-ext || true
    else
        echo "warning: ${DEVICE} not present; replay mode still works"
    fi
fi

step "Build"
cmake -S "${RUNTIME_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT="${ORT_ROOT}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"

step "Unit tests"
ctest --test-dir "${BUILD_DIR}" --output-on-failure

if [[ -e "${DEVICE}" ]]; then
    step "Camera capture test (${DEVICE})"
    EDGE_AI_V4L2_DEVICE="${DEVICE}" "${BUILD_DIR}/tests/v4l2_source_test" || echo "warning: camera capture test failed"
fi

step "Model load smoke test"
if [[ -f "${MODEL}" ]]; then
    "${BUILD_DIR}/app/edge_ai_runtime" \
        --source "replay:${RUNTIME_DIR}/tests/data/test.mov" \
        --model "${MODEL}" \
        --mode always --sync --max-frames 30 --no-overlay \
        --output "${REPO_DIR}/runs/board-smoke"
    grep -E '"(load_ms|average_inference_ms|p95_inference_ms|peak_ram_mb)"' "${REPO_DIR}/runs/board-smoke/summary.json"
else
    echo "warning: ${MODEL} missing; copy the exported ONNX model to the board"
fi

step "Done"
echo "Live camera:  ${BUILD_DIR}/app/edge_ai_runtime --config ${RUNTIME_DIR}/config/runtime.json --source v4l2:${DEVICE} --model ${MODEL}"
