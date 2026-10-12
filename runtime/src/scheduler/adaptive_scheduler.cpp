#include <edge_ai/scheduler/adaptive_scheduler.hpp>

#include <cstring>
#include <stdexcept>

const char* toString(Action action){
    switch(action){
        case Action::IDLE: return "IDLE";
        case Action::TRACK_ONLY: return "TRACK_ONLY";
        case Action::RUN_DETECTOR: return "RUN_DETECTOR";
    }
    return "?";
}

const char* toString(RuntimeState state){
    switch(state){
        case RuntimeState::IDLE: return "IDLE";
        case RuntimeState::RECHECK: return "RECHECK";
        case RuntimeState::ACTIVE: return "ACTIVE";
    }
    return "?";
}

const char* toString(SchedulerMode mode){
    switch(mode){
        case SchedulerMode::ALWAYS: return "always";
        case SchedulerMode::FIXED_INTERVAL: return "fixed_interval";
        case SchedulerMode::MOTION_GATED: return "motion_gated";
        case SchedulerMode::ADAPTIVE: return "adaptive";
    }
    return "?";
}

SchedulerMode schedulerModeFromString(const std::string& mode){
    if(mode == "always") return SchedulerMode::ALWAYS;
    if(mode == "fixed_interval") return SchedulerMode::FIXED_INTERVAL;
    if(mode == "motion_gated") return SchedulerMode::MOTION_GATED;
    if(mode == "adaptive") return SchedulerMode::ADAPTIVE;
    throw std::invalid_argument("unknown scheduler mode: " + mode);
}

AdaptiveScheduler::AdaptiveScheduler(const SchedulerConfig& config)
    : config_(config)
{
}

void AdaptiveScheduler::reset(){
    state_ = RuntimeState::IDLE;
    frames_seen_ = 0;
    last_ts_ns_.reset();
    last_run_ns_.reset();
    last_activity_ns_.reset();
}

bool AdaptiveScheduler::elapsed(uint64_t now_ns, uint32_t interval_ms) const{
    return !last_run_ns_.has_value() ||
        now_ns < *last_run_ns_ ||
        now_ns - *last_run_ns_ >= static_cast<uint64_t>(interval_ms)*1'000'000ULL;
}

Decision AdaptiveScheduler::run(const char* reason){
    // Any semantic check out of IDLE is a recheck until the detector answers.
    if(state_ == RuntimeState::IDLE) state_ = RuntimeState::RECHECK;
    return {Action::RUN_DETECTOR, reason, state_};
}

Decision AdaptiveScheduler::skip(Action action, const char* reason){
    return {action, reason, state_};
}

void AdaptiveScheduler::markDetectorRun(uint64_t ts_ns){
    last_run_ns_ = ts_ns;
}

void AdaptiveScheduler::onDetectorResult(size_t detections, uint64_t ts_ns){
    if(detections > 0){
        state_ = RuntimeState::ACTIVE;
        last_activity_ns_ = ts_ns;
    }
    else if(state_ == RuntimeState::RECHECK){
        // No person found: go back to idle. Continued motion will request a new recheck.
        state_ = RuntimeState::IDLE;
    }
}

Decision AdaptiveScheduler::decide(const SchedulerInputs& in){
    // A restarted/looped stream moves time backwards: start over.
    if(last_ts_ns_.has_value() && in.ts_ns < *last_ts_ns_) reset();
    last_ts_ns_ = in.ts_ns;
    ++frames_seen_;

    if(in.motion_active || in.confirmed_tracks > 0) last_activity_ns_ = in.ts_ns;

    // ACTIVE -> IDLE once people are gone and the scene settled for active_hold_ms.
    if(state_ == RuntimeState::ACTIVE && in.confirmed_tracks == 0 && !in.motion_active){
        const uint64_t hold_ns = static_cast<uint64_t>(config_.active_hold_ms)*1'000'000ULL;
        if(!last_activity_ns_.has_value() || in.ts_ns - *last_activity_ns_ >= hold_ns){
            state_ = RuntimeState::IDLE;
        }
    }

    const bool startup = frames_seen_ <= config_.startup_frames;

    Decision decision;
    switch(config_.mode){
        case SchedulerMode::ALWAYS:
            return run("always");

        case SchedulerMode::FIXED_INTERVAL:
            if(startup) return run("startup");
            decision = elapsed(in.ts_ns, config_.fixed_interval_ms)
                ? run("fixed_interval")
                : skip(Action::TRACK_ONLY, "track");
            break;

        case SchedulerMode::MOTION_GATED:{
            if(startup) return run("startup");
            const uint64_t hold_ns = static_cast<uint64_t>(config_.active_hold_ms)*1'000'000ULL;
            const bool scene_active = in.motion_active ||
                (last_activity_ns_.has_value() && in.ts_ns - *last_activity_ns_ < hold_ns);
            if(scene_active){
                decision = elapsed(in.ts_ns, config_.fixed_interval_ms)
                    ? run("fixed_interval")
                    : skip(Action::TRACK_ONLY, "track");
            }
            else{
                decision = elapsed(in.ts_ns, config_.idle_poll_ms)
                    ? run("idle_poll")
                    : skip(Action::IDLE, "idle");
            }
            break;
        }

        case SchedulerMode::ADAPTIVE:
            if(startup) return run("startup");
            decision = decideAdaptive(in);
            break;
    }

    // Detector busy: drop non-urgent periodic checks instead of preprocessing a frame
    // that would only replace (or be refused by) the pending request.
    if(decision.action == Action::RUN_DETECTOR && in.detector_busy){
        const char* r = decision.reason;
        const bool periodic =
            std::strcmp(r, "fixed_interval") == 0 ||
            std::strcmp(r, "idle_poll") == 0 ||
            std::strcmp(r, "max_semantic_age") == 0 ||
            std::strcmp(r, "confirm_track") == 0;
        if(periodic){
            const Action fallback = in.confirmed_tracks > 0 || in.motion_active ? Action::TRACK_ONLY : Action::IDLE;
            return skip(fallback, "detector_busy");
        }
    }
    return decision;
}

Decision AdaptiveScheduler::decideAdaptive(const SchedulerInputs& in){
    if(in.tracker_failed) return run("tracker_failed");

    if(in.confirmed_tracks == 0){
        if(in.motion_event) return run("motion_event");

        if(in.motion_active){
            return elapsed(in.ts_ns, config_.min_semantic_interval_ms)
                ? run("motion_no_track")
                : skip(Action::TRACK_ONLY, "await_result");
        }

        if(in.tentative_tracks > 0 && elapsed(in.ts_ns, config_.min_semantic_interval_ms)){
            return run("confirm_track");
        }

        return elapsed(in.ts_ns, config_.idle_poll_ms)
            ? run("idle_poll")
            : skip(Action::IDLE, "idle");
    }

    if(in.motion_active &&
       in.unexplained_motion_fraction >= config_.unexplained_motion_fraction &&
       elapsed(in.ts_ns, config_.min_semantic_interval_ms)){
        return run("unexplained_motion");
    }

    if(in.abrupt_motion || in.tracker_weak){
        if(elapsed(in.ts_ns, config_.min_semantic_interval_ms)){
            return run(in.abrupt_motion ? "abrupt_motion" : "tracker_weak");
        }
        return skip(Action::TRACK_ONLY, "track");
    }

    if(elapsed(in.ts_ns, config_.max_semantic_age_ms)) return run("max_semantic_age");

    return skip(Action::TRACK_ONLY, "track");
}
