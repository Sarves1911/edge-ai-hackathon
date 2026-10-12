#!/usr/bin/env bash
# Runs the R0-R6 benchmark ladder on identical recordings and prints the scorecard.
#
#   runtime/bench/run_ladder.sh [options] recording.mp4 [more recordings...]
#
#   --rungs "0 1 2 3 4 5 6"   which rungs to run (default all)
#   --out runs/ladder         output root
#   --calibration clip.mp4    INT8 calibration clip (default: first recording; prefer a separate split)
#   --realtime                pace replay at the recorded frame rate (use for power/CPU measurement)
#   --threads N               ONNX Runtime intra-op threads for R1-R6
#   --skip-prepare            do not export/quantize missing models
#
# R0 is the Python reference (PyTorch FP32, 640) through the existing edge-detect CLI.
# R1-R6 are the C++ runtime with runtime/config/ladder/rN.json.
set -euo pipefail

RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${RUNTIME_DIR}/.." && pwd)"
BIN="${RUNTIME_DIR}/build/app/edge_ai_runtime"
MODELS="${REPO_DIR}/models"

RUNGS="0 1 2 3 4 5 6"
OUT="${REPO_DIR}/runs/ladder"
CALIBRATION=""
EXTRA=()
PREPARE=1
CLIPS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rungs) RUNGS="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --calibration) CALIBRATION="$2"; shift 2 ;;
        --realtime) EXTRA+=(--realtime); shift ;;
        --threads) EXTRA+=(--threads "$2"); shift 2 ;;
        --skip-prepare) PREPARE=0; shift ;;
        -h|--help) sed -n 2,16p "$0"; exit 0 ;;
        *) CLIPS+=("$1"); shift ;;
    esac
done

[[ ${#CLIPS[@]} -gt 0 ]] || { echo "usage: $0 [options] recording..." >&2; exit 2; }
[[ -x "${BIN}" ]] || { echo "build the runtime first: cmake --build runtime/build" >&2; exit 1; }

HAVE_UV=0
command -v uv >/dev/null && HAVE_UV=1

edge_detect(){ (cd "${REPO_DIR}" && uv run --extra export edge-detect "$@"); }

export_model(){   # export_model <height> <width> <output>
    local tmp
    tmp="$(mktemp -d)"
    (cd "${tmp}" && cp "${MODELS}/yolov8n.pt" . 2>/dev/null || true
     cd "${tmp}" && uv run --project "${REPO_DIR}" --extra export edge-detect export \
        --model yolov8n.pt --format onnx --imgsz "$1" "$2" >/dev/null)
    mv "${tmp}/yolov8n.onnx" "$3"
    rm -rf "${tmp}"
}

if [[ ${PREPARE} -eq 1 && ${HAVE_UV} -eq 1 ]]; then
    mkdir -p "${MODELS}"
    [[ -f "${MODELS}/yolov8n.onnx" ]] || { echo "exporting yolov8n.onnx (256x320)"; export_model 256 320 "${MODELS}/yolov8n.onnx"; }
    [[ -f "${MODELS}/yolov8n-640.onnx" ]] || { echo "exporting yolov8n-640.onnx"; export_model 640 640 "${MODELS}/yolov8n-640.onnx"; }
    if [[ ! -f "${MODELS}/yolov8n-int8.onnx" ]]; then
        echo "quantizing yolov8n-int8.onnx"
        (cd "${REPO_DIR}" && uv run --extra export python tools/model/quantize_onnx.py \
            --model "${MODELS}/yolov8n.onnx" --source "${CALIBRATION:-${CLIPS[0]}}" \
            --samples 128 --output "${MODELS}/yolov8n-int8.onnx")
    fi
fi

peak_rss(){   # runs a command under /usr/bin/time, writing peak RSS (MB) next to its output
    local out_dir="$1"; shift
    if [[ "$(uname -s)" == "Darwin" ]]; then
        /usr/bin/time -l "$@" 2> >(tee "${out_dir}/time.txt" >&2)
        awk '/maximum resident set size/ {printf "%.1f\n", $1/1048576}' "${out_dir}/time.txt" > "${out_dir}/peak_rss_mb.txt"
    elif [[ -x /usr/bin/time ]]; then
        /usr/bin/time -v "$@" 2> >(tee "${out_dir}/time.txt" >&2)
        awk -F: '/Maximum resident set size/ {printf "%.1f\n", $2/1024}' "${out_dir}/time.txt" > "${out_dir}/peak_rss_mb.txt"
    else
        "$@"
    fi
}

for clip in "${CLIPS[@]}"; do
    name="$(basename "${clip%.*}")"
    clip_abs="$(cd "$(dirname "${clip}")" && pwd)/$(basename "${clip}")"

    for rung in ${RUNGS}; do
        dir="${OUT}/${name}/r${rung}"
        rm -rf "${dir}"
        mkdir -p "${dir}"
        echo "=== ${name} R${rung}"

        if [[ "${rung}" == "0" ]]; then
            if [[ ${HAVE_UV} -eq 0 ]]; then echo "skip R0: uv/Python reference not available"; continue; fi
            peak_rss "${dir}" bash -c "cd '${REPO_DIR}' && uv run --extra export edge-detect run \
                --source '${clip_abs}' --model '${MODELS}/yolov8n.pt' --imgsz 640 640 \
                --device cpu --mode always --no-annotated --output '${dir}'" >/dev/null
            continue
        fi

        config="${RUNTIME_DIR}/config/ladder/r${rung}.json"
        model="$(sed -n 's/.*"path": *"\([^"]*\)".*/\1/p' "${config}" | head -1)"
        if [[ ! -f "${REPO_DIR}/${model}" ]]; then
            echo "skip R${rung}: ${model} missing"
            continue
        fi

        (cd "${REPO_DIR}" && "${BIN}" --config "${config}" --source "replay:${clip_abs}" \
            --output "${dir}" "${EXTRA[@]+"${EXTRA[@]}"}" 2>&1 | grep -v swscaler)
    done
done

python3 "${RUNTIME_DIR}/bench/summarize.py" "${OUT}" | tee "${OUT}/scorecard.md"
