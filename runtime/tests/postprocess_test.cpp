#include <cstddef>
#include <iostream>
#include <vector>

#include <edge_ai/detector/postprocess.hpp>

#include "test_common.hpp"

namespace {

// Synthetic YOLOv8 output: [num_attrs = 4 + classes, N] attribute-major.
struct RawOutput{
    size_t classes;
    size_t candidates;
    std::vector<float> data;

    RawOutput(size_t classes_, size_t candidates_)
        : classes(classes_), candidates(candidates_), data((4 + classes_)*candidates_, 0.0f) {}

    void set(size_t i, float cx, float cy, float w, float h, size_t cls, float score){
        data[0*candidates + i] = cx;
        data[1*candidates + i] = cy;
        data[2*candidates + i] = w;
        data[3*candidates + i] = h;
        data[(4 + cls)*candidates + i] = score;
    }

    size_t attrs() const { return 4 + classes; }
};

PreprocessMetadata identity(size_t w, size_t h){
    return calculateLetterboxGeometry(w, h, w, h);
}

} // namespace

int main(){
    PostprocessConfig config;   // conf 0.25, iou 0.45, class 0

    // Box decode: cx/cy/w/h -> xyxy.
    {
        RawOutput out(3, 1);
        out.set(0, 50, 40, 20, 10, 0, 0.9f);
        const auto dets = postprocessYolov8(out.data.data(), out.attrs(), out.candidates, config, identity(100, 100));
        CHECK(dets.size() == 1);
        CHECK_NEAR(dets[0].x1, 40, 1e-4);
        CHECK_NEAR(dets[0].y1, 35, 1e-4);
        CHECK_NEAR(dets[0].x2, 60, 1e-4);
        CHECK_NEAR(dets[0].y2, 45, 1e-4);
        CHECK_NEAR(dets[0].score, 0.9f, 1e-6);
    }

    // Person-class filtering: a confident non-person must not appear.
    {
        RawOutput out(3, 2);
        out.set(0, 20, 20, 10, 10, 2, 0.95f);   // class 2 (car)
        out.set(1, 70, 70, 10, 10, 0, 0.60f);   // person
        const auto dets = postprocessYolov8(out.data.data(), out.attrs(), out.candidates, config, identity(100, 100));
        CHECK(dets.size() == 1);
        CHECK_NEAR(dets[0].x1, 65, 1e-4);
    }

    // Confidence filtering at the threshold.
    {
        RawOutput out(1, 3);
        out.set(0, 10, 10, 4, 4, 0, 0.24f);
        out.set(1, 30, 30, 4, 4, 0, 0.25f);
        out.set(2, 60, 60, 4, 4, 0, 0.80f);
        const auto dets = postprocessYolov8(out.data.data(), out.attrs(), out.candidates, config, identity(100, 100));
        CHECK(dets.size() == 2);
        CHECK(dets[0].score > dets[1].score);
    }

    // NMS: overlapping duplicates collapse to the best; distant box survives.
    {
        RawOutput out(1, 4);
        out.set(0, 50, 50, 40, 40, 0, 0.70f);
        out.set(1, 52, 51, 40, 40, 0, 0.90f);   // best, overlaps 0 and 2
        out.set(2, 48, 49, 40, 40, 0, 0.50f);
        out.set(3, 150, 150, 40, 40, 0, 0.40f); // separate person
        const auto dets = postprocessYolov8(out.data.data(), out.attrs(), out.candidates, config, identity(200, 200));
        CHECK(dets.size() == 2);
        CHECK_NEAR(dets[0].score, 0.90f, 1e-6);
        CHECK_NEAR(dets[1].score, 0.40f, 1e-6);
    }

    // NMS keeps boxes whose IoU is below the threshold.
    {
        std::vector<Detection> boxes = {
            {0, 0, 10, 10, 0.9f},
            {6, 0, 16, 10, 0.8f}   // IoU = 40/160 = 0.25
        };
        CHECK_NEAR(boxIou(boxes[0], boxes[1]), 0.25f, 1e-5);
        CHECK(nonMaxSuppression(boxes, 0.45f, 300).size() == 2);
        CHECK(nonMaxSuppression(boxes, 0.20f, 300).size() == 1);
        CHECK(nonMaxSuppression(boxes, 0.45f, 1).size() == 1);
    }

    // Undo letterbox: 640x480 frame in a 320x256 model (scale .5, pad_top 8).
    {
        const PreprocessMetadata meta = calculateLetterboxGeometry(640, 480, 320, 256);
        RawOutput out(1, 1);
        // Original box (100, 200)-(300, 400) -> model (50, 108)-(150, 208)
        out.set(0, 100, 158, 100, 100, 0, 0.8f);
        const auto dets = postprocessYolov8(out.data.data(), out.attrs(), out.candidates, config, meta);
        CHECK(dets.size() == 1);
        CHECK_NEAR(dets[0].x1, 100, 1e-3);
        CHECK_NEAR(dets[0].y1, 200, 1e-3);
        CHECK_NEAR(dets[0].x2, 300, 1e-3);
        CHECK_NEAR(dets[0].y2, 400, 1e-3);
    }

    // Clipping: boxes extending into padding / past the edge clamp to the image.
    {
        const PreprocessMetadata meta = calculateLetterboxGeometry(640, 480, 320, 256);
        RawOutput out(1, 1);
        out.set(0, 10, 10, 40, 40, 0, 0.8f);   // model (-10,-10)-(30,30) reaches into top padding
        const auto dets = postprocessYolov8(out.data.data(), out.attrs(), out.candidates, config, meta);
        CHECK(dets.size() == 1);
        CHECK(dets[0].x1 == 0.0f);
        CHECK(dets[0].y1 == 0.0f);
        CHECK(dets[0].x2 <= 640.0f && dets[0].y2 <= 480.0f);
    }

    // Empty output yields no detections.
    {
        RawOutput out(80, 10);
        const auto dets = postprocessYolov8(out.data.data(), out.attrs(), out.candidates, config, identity(64, 64));
        CHECK(dets.empty());
    }

    std::cout << "All postprocess tests passed.\n";
    return 0;
}
