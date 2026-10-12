#!/usr/bin/env bash
# Builds the runtime and runs every CTest under linux/arm64 (Docker + QEMU on x86/macOS hosts).
#
#   runtime/scripts/build_arm64.sh            configure, build, ctest
#   runtime/scripts/build_arm64.sh --shell    open a shell in the ARM64 environment
set -euo pipefail

RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${RUNTIME_DIR}/.." && pwd)"
IMAGE="edge-ai-runtime-arm64"
BUILD_DIR="runtime/build-arm64"

docker buildx build \
    --platform linux/arm64 \
    --load \
    -t "${IMAGE}" \
    -f "${RUNTIME_DIR}/docker/Dockerfile.arm64" \
    "${RUNTIME_DIR}/docker"

run=(docker run --rm --platform linux/arm64 -v "${REPO_DIR}:/src" -w /src)

if [[ "${1:-}" == "--shell" ]]; then
    exec "${run[@]}" -it "${IMAGE}" bash
fi

"${run[@]}" "${IMAGE}" bash -c "
    set -euo pipefail
    uname -m
    cmake -S runtime -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release
    cmake --build ${BUILD_DIR} -j\$(nproc)
    ctest --test-dir ${BUILD_DIR} --output-on-failure
    file ${BUILD_DIR}/app/edge_ai_runtime 2>/dev/null || true
"
