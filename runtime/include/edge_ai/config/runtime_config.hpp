#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <edge_ai/detector/detector_worker.hpp>
#include <edge_ai/motion/motion_gate.hpp>
#include <edge_ai/scheduler/adaptive_scheduler.hpp>
#include <edge_ai/tracker/tracker.hpp>

struct ModelConfig{
    std::string path = "models/yolov8n.onnx";
    float confidence = 0.25f;
    float iou = 0.45f;
    size_t class_id = 0;
    int threads = 0;              // 0 = ONNX Runtime default

    // Used by the simulated detector; the ONNX model's own input shape wins otherwise.
    size_t input_width = 320;
    size_t input_height = 256;
};

struct CameraConfig{
    size_t width = 640;
    size_t height = 480;
    uint32_t fps = 30;
    uint32_t buffers = 4;
    int poll_timeout_ms = 1000;
};

struct PipelineConfig{
    std::string source;               // replay:<path> | v4l2:/dev/videoN | <path>
    bool realtime = false;            // replay pacing
    bool async = true;
    BusyPolicy busy_policy = BusyPolicy::LATEST;
    std::string detector = "onnx";    // onnx | simulated
    std::string simulated_jsonl;      // cached detections for the simulated detector
    uint32_t simulated_delay_ms = 0;
    uint64_t max_frames = 0;          // 0 = whole input
};

struct OutputConfig{
    std::string directory = "runs/cpp";
    bool save_frames_csv = true;
    bool save_jsonl = true;
    bool overlay = true;
    uint32_t overlay_interval_ms = 1000;
    uint32_t viewer_port = 0;          // 0 = disabled
    uint32_t viewer_fps = 5;
};

struct RuntimeConfig{
    ModelConfig model;
    MotionConfig motion;
    TrackerConfig tracker;
    SchedulerConfig scheduler;
    PipelineConfig pipeline;
    CameraConfig camera;
    OutputConfig output;
};

// Missing keys keep their defaults. Unknown keys and invalid values throw
// std::invalid_argument so threshold typos cannot silently fall back to defaults.
RuntimeConfig parseConfig(const std::string& json_text);
RuntimeConfig loadConfig(const std::string& path);

void validateConfig(const RuntimeConfig& config);
std::string configToJson(const RuntimeConfig& config);
