#pragma once

#include <cstddef>
#include <vector>

#include <edge_ai/detector/detection.hpp>
#include <edge_ai/preprocess/preprocess.hpp>

struct PostprocessConfig{
    float confidence = 0.25f;
    float iou = 0.45f;
    size_t class_id = 0;   // COCO person
    size_t max_detections = 300;
};

float boxIou(const Detection& a, const Detection& b);

// Greedy NMS; returns surviving boxes sorted by descending score.
std::vector<Detection> nonMaxSuppression(std::vector<Detection> boxes, float iou_threshold, size_t max_detections);

// Decodes a YOLOv8 [1, num_attrs, num_candidates] attribute-major output
// (cx, cy, w, h, class scores...) into person detections in original-frame coordinates.
std::vector<Detection> postprocessYolov8(
    const float* output,
    size_t num_attrs,
    size_t num_candidates,
    const PostprocessConfig& config,
    const PreprocessMetadata& meta
);
