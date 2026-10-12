#pragma once

#include "edge_ai/core/frame.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

// YOLO letterbox padding value (matches Ultralytics and tools/model/quantize_onnx.py).
constexpr uint8_t kLetterboxPad = 114;

struct PreprocessMetadata{
    float scale;
    uint16_t pad_left;
    uint16_t pad_top;
    uint16_t original_height;
    uint16_t original_width;
    uint16_t target_height;
    uint16_t target_width;
    uint16_t resized_height;
    uint16_t resized_width;
};

struct PreprocessResult{
    std::vector<float> tensor;
    PreprocessMetadata meta;
};

PreprocessMetadata calculateLetterboxGeometry(size_t width, size_t height, size_t target_width, size_t target_height);

// Writes a 3 x target_height x target_width float32 CHW tensor (RGB, 0..1) into dst_chw.
// dst_chw must hold 3 * meta.target_width * meta.target_height floats. No allocation.
void preprocessInto(const ImageView& view, const PreprocessMetadata& meta, float* dst_chw);

PreprocessResult preprocess(const ImageView& view, size_t target_width, size_t target_height);

// Maps a point from model (letterboxed) space back to original-frame pixel space,
// clamped to the original image bounds.
void modelToOriginal(float& x, float& y, const PreprocessMetadata& meta);
