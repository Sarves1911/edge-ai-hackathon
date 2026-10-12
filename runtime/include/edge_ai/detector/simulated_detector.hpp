#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <edge_ai/detector/detector.hpp>

// Deterministic stand-in for the real detector.
//
// Returns scripted/cached detections for a source timestamp and can sleep to
// emulate expensive inference (latency injection).
class SimulatedDetector : public Detector{
    public:
        using Script = std::function<std::vector<Detection>(uint64_t frame_ts_ns)>;
        using Timeline = std::vector<std::pair<uint64_t, std::vector<Detection>>>;

        SimulatedDetector(size_t input_width, size_t input_height, Script script, uint32_t delay_ms = 0);

        // Script returning the timeline entry nearest to each requested timestamp.
        static Script timelineScript(Timeline timeline);

        // Loads cached detections from a Python reference / C++ detections.jsonl
        // (timestamp_ms + detections[].bbox_xyxy/confidence, person class only).
        static Timeline loadJsonl(const std::string& path);

        size_t inputWidth() const override { return input_width_; }
        size_t inputHeight() const override { return input_height_; }

        DetectionBatch infer(const float* chw, const PreprocessMetadata& meta, uint64_t frame_ts_ns, uint64_t frame_id) override;

        void setDelayMs(uint32_t delay_ms) { delay_ms_ = delay_ms; }
        size_t calls() const { return calls_.load(); }

    private:
        size_t input_width_;
        size_t input_height_;
        Script script_;
        std::atomic<uint32_t> delay_ms_;
        std::atomic<size_t> calls_{0};
};
