#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

enum class Action{
    IDLE,
    TRACK_ONLY,
    RUN_DETECTOR
};

enum class RuntimeState{
    IDLE,
    RECHECK,
    ACTIVE
};

enum class SchedulerMode{
    ALWAYS,          // R0-R3: detector on every frame
    FIXED_INTERVAL,  // R4: detector every fixed_interval_ms, track in between
    MOTION_GATED,    // R5: fixed interval while scene is active, idle while quiet
    ADAPTIVE         // R6: motion + tracker + semantic age decide
};

const char* toString(Action action);
const char* toString(RuntimeState state);
const char* toString(SchedulerMode mode);
SchedulerMode schedulerModeFromString(const std::string& mode);

struct SchedulerConfig{
    SchedulerMode mode = SchedulerMode::ADAPTIVE;

    uint32_t startup_frames = 3;
    uint32_t fixed_interval_ms = 500;

    // Safety poll while idle so a missed motion event cannot suppress inference forever.
    uint32_t idle_poll_ms = 2000;

    // Stay ACTIVE this long after the last motion/track before returning to IDLE.
    uint32_t active_hold_ms = 1500;

    // Upper bound on time between semantic checks while people are tracked.
    uint32_t max_semantic_age_ms = 1000;

    // Shortened semantic interval when motion is abrupt or tracks are weak.
    uint32_t min_semantic_interval_ms = 200;

    // Fraction of changed pixels outside tracked boxes that counts as unexplained motion.
    float unexplained_motion_fraction = 0.35f;
};

struct SchedulerInputs{
    uint64_t ts_ns = 0;

    bool motion_active = false;
    bool motion_event = false;
    float motion_score = 0.0f;
    float unexplained_motion_fraction = 0.0f;

    size_t confirmed_tracks = 0;
    size_t tentative_tracks = 0;
    bool tracker_failed = false;
    bool tracker_weak = false;
    bool abrupt_motion = false;

    bool detector_busy = false;
};

struct Decision{
    Action action = Action::IDLE;
    const char* reason = "";
    RuntimeState state = RuntimeState::IDLE;
};

// Pure policy: no threads, no detector. Decides when semantic inference is justified.
class AdaptiveScheduler{
    public:
        explicit AdaptiveScheduler(const SchedulerConfig& config = SchedulerConfig{});

        Decision decide(const SchedulerInputs& inputs);

        // Called when a detector request was actually accepted by the worker.
        void markDetectorRun(uint64_t ts_ns);

        // Called when a detector result arrives (drives RECHECK -> ACTIVE/IDLE).
        void onDetectorResult(size_t detections, uint64_t ts_ns);

        RuntimeState state() const { return state_; }
        std::optional<uint64_t> lastDetectorTs() const { return last_run_ns_; }

        void reset();

    private:
        Decision run(const char* reason);
        Decision skip(Action action, const char* reason);
        bool elapsed(uint64_t now_ns, uint32_t interval_ms) const;

        Decision decideAdaptive(const SchedulerInputs& in);

        SchedulerConfig config_;
        RuntimeState state_ = RuntimeState::IDLE;

        uint64_t frames_seen_ = 0;
        std::optional<uint64_t> last_ts_ns_;
        std::optional<uint64_t> last_run_ns_;
        std::optional<uint64_t> last_activity_ns_;
};
