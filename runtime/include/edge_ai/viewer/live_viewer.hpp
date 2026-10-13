#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <edge_ai/core/frame.hpp>
#include <edge_ai/detector/detection.hpp>

struct ViewerStatus{
    uint64_t frame_id = 0;
    std::string state;
    std::string action;
    std::string reason;
    size_t persons = 0;
    size_t detector_calls = 0;
    size_t frames = 0;
    float last_result_age_ms = -1.0f;
    float last_infer_ms = 0.0f;
    float motion_score = 0.0f;
    double fps = 0.0;
    std::vector<Detection> detections;
    std::vector<int> track_ids;
};

// Small, dependency-light HTTP server for the live demo. Frames are converted
// to JPEG and annotated before they leave the board; a browser only needs to
// poll /frame.jpg and /state.json.
class LiveViewer{
    public:
        // Port 0 asks the OS for a free port (useful for tests).
        LiveViewer(uint16_t port, uint32_t max_fps = 5);
        ~LiveViewer();

        LiveViewer(const LiveViewer&) = delete;
        LiveViewer& operator=(const LiveViewer&) = delete;
        LiveViewer(LiveViewer&&) = delete;
        LiveViewer& operator=(LiveViewer&&) = delete;

        // ImageView remains caller-owned and is consumed before this returns.
        void publish(const ImageView& image, const ViewerStatus& status);

        uint16_t port() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
};
