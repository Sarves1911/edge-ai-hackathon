#pragma once

#include <cstdint>
#include <vector>

// Person detection in ORIGINAL frame pixel coordinates.
// (x1, y1) is top-left, (x2, y2) is bottom-right.
struct Detection{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
};

struct DetectionBatch{
    // Timestamp of the SOURCE frame that produced these detections,
    // not the time the detector finished.
    uint64_t frame_ts_ns = 0;
    uint64_t frame_id = 0;

    // Owned, so tracker/scheduler code never depends on detector buffers.
    std::vector<Detection> detections;

    float infer_ms = 0.0f;
    float postprocess_ms = 0.0f;
};
