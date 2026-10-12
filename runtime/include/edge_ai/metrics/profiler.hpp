#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <edge_ai/detector/detection.hpp>

// Linear-interpolation percentile (same definition as the Python reference metrics).
float percentile(std::vector<float> values, float percent);

struct ResourceUsage{
    double cpu_user_s = 0.0;
    double cpu_system_s = 0.0;
    double peak_rss_mb = 0.0;
};

ResourceUsage sampleResourceUsage();

struct FrameRecord{
    uint64_t frame_id = 0;
    uint64_t ts_ns = 0;

    float motion_ms = 0.0f;
    float motion_score = 0.0f;
    bool motion_active = false;
    bool motion_event = false;

    float preprocess_ms = 0.0f;   // only when a request was submitted
    float track_ms = 0.0f;
    float total_ms = 0.0f;        // control-loop work for this frame

    std::string action;
    std::string reason;
    std::string state;
    std::string submit;           // accepted | replaced | skipped | "" when not requested

    size_t tracks = 0;
    size_t confirmed_tracks = 0;
    float result_age_ms = -1.0f;  // source-time age of the latest applied detector result
};

struct DetectorRecord{
    uint64_t frame_id = 0;
    uint64_t frame_ts_ns = 0;
    float infer_ms = 0.0f;
    float postprocess_ms = 0.0f;
    float result_latency_ms = 0.0f;   // submit -> result consumed, wall clock
    size_t detections = 0;
};

// Streams per-frame CSV and detection JSONL so long soak runs keep bounded memory;
// only scalar timing samples are retained for the summary percentiles.
class Profiler{
    public:
        Profiler() = default;

        void openOutputs(const std::string& directory, bool frames_csv, bool detections_jsonl);
        void start();

        void addFrame(const FrameRecord& record);
        void addDetectorResult(const DetectorRecord& record);

        // Python-compatible JSONL line (frame_index, timestamp_ms, detections[...]).
        void writeDetections(
            uint64_t frame_id,
            uint64_t ts_ns,
            size_t width,
            size_t height,
            bool ran,
            const std::string& reason,
            float result_age_ms,
            const std::vector<Detection>& detections,
            const std::vector<int>& track_ids
        );

        void setModelInfo(const std::string& path, double size_mb, float load_ms);
        void setExtra(const std::string& key, double value);

        size_t frames() const { return frames_; }
        size_t detectorCalls() const { return infer_ms_.size(); }

        // Summary as a JSON object string.
        std::string summaryJson() const;
        void writeSummary(const std::string& path) const;

        void flush();

    private:
        std::ofstream frames_csv_;
        std::ofstream detections_jsonl_;

        uint64_t started_ns_ = 0;
        uint64_t first_ts_ns_ = 0;
        uint64_t last_ts_ns_ = 0;
        size_t frames_ = 0;

        std::vector<float> infer_ms_;
        std::vector<float> postprocess_ms_;
        std::vector<float> result_latency_ms_;
        std::vector<float> preprocess_ms_;
        std::vector<float> frame_ms_;
        std::vector<float> motion_ms_;
        std::vector<float> result_age_ms_;

        size_t motion_active_frames_ = 0;
        size_t motion_events_ = 0;
        size_t total_detections_ = 0;
        std::map<std::string, size_t> reasons_;
        std::map<std::string, size_t> actions_;
        std::map<std::string, size_t> states_;
        std::map<std::string, size_t> submits_;
        std::map<std::string, double> extra_;

        std::string model_path_;
        double model_size_mb_ = 0.0;
        float model_load_ms_ = 0.0f;
};
