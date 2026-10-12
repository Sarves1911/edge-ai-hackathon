#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>

#include <edge_ai/capture/frame_source.hpp>
#include <edge_ai/config/runtime_config.hpp>
#include <edge_ai/detector/detector.hpp>
#include <edge_ai/detector/detector_worker.hpp>
#include <edge_ai/metrics/profiler.hpp>
#include <edge_ai/motion/motion_gate.hpp>
#include <edge_ai/scheduler/adaptive_scheduler.hpp>
#include <edge_ai/tracker/tracker.hpp>

struct PipelineStatus{
    uint64_t frame_id = 0;
    uint64_t ts_ns = 0;
    RuntimeState state = RuntimeState::IDLE;
    Action action = Action::IDLE;
    const char* reason = "";
    size_t persons = 0;
    size_t detector_calls = 0;
    size_t frames = 0;
    float last_result_age_ms = -1.0f;
    float last_infer_ms = 0.0f;
    float motion_score = 0.0f;
};

// The capture/control loop: motion gate -> tracker predict -> scheduler -> async
// detector submit, plus timestamp-aware consumption of finished detector results.
//
// Component logic lives in the modules; this class only wires them together.
class Pipeline{
    public:
        Pipeline(const RuntimeConfig& config, Detector& detector, Profiler& profiler);

        // Processes one frame. The frame's pixels are only read during this call.
        PipelineStatus step(const Frame& frame);

        // Runs until end of stream, max_frames, or *stop becomes true.
        size_t run(
            FrameSource& source,
            uint64_t max_frames = 0,
            const std::atomic<bool>* stop = nullptr,
            const std::function<void(const PipelineStatus&)>& on_frame = {}
        );

        // Waits for in-flight inference and applies its result.
        void drain();

        const Tracker& tracker() const { return tracker_; }
        const AdaptiveScheduler& scheduler() const { return scheduler_; }
        const DetectorWorker& worker() const { return worker_; }
        const PipelineStatus& status() const { return status_; }

    private:
        void consumeResults();
        bool motionEnabled() const;
        bool trackingEnabled() const;

        RuntimeConfig config_;
        Profiler& profiler_;

        MotionGate motion_;
        Tracker tracker_;
        AdaptiveScheduler scheduler_;
        DetectorWorker worker_;

        // Submit wall time per frame id, for result latency (at most a few entries).
        std::map<uint64_t, uint64_t> submit_wall_ns_;

        std::optional<DetectionBatch> last_batch_;
        bool batch_arrived_this_frame_ = false;
        PipelineStatus status_;
};
