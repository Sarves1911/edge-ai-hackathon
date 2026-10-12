#include <cstring>
#include <iostream>

#include <edge_ai/scheduler/adaptive_scheduler.hpp>

#include "test_common.hpp"

namespace {

constexpr uint64_t kFrame = 33*kMs;

SchedulerConfig adaptiveConfig(){
    SchedulerConfig c;
    c.mode = SchedulerMode::ADAPTIVE;
    c.startup_frames = 0;
    return c;
}

SchedulerInputs quiet(uint64_t ts){
    SchedulerInputs in;
    in.ts_ns = ts;
    return in;
}

bool is(const Decision& d, Action action, const char* reason){
    return d.action == action && std::strcmp(d.reason, reason) == 0;
}

// Feeds a decision back like the pipeline does when the worker accepts a request.
Decision decideAndRun(AdaptiveScheduler& s, const SchedulerInputs& in){
    const Decision d = s.decide(in);
    if(d.action == Action::RUN_DETECTOR) s.markDetectorRun(in.ts_ns);
    return d;
}

} // namespace

int main(){
    // No tracks + no motion -> IDLE (after one initial idle poll).
    {
        AdaptiveScheduler s(adaptiveConfig());
        CHECK(is(decideAndRun(s, quiet(0)), Action::RUN_DETECTOR, "idle_poll"));
        s.onDetectorResult(0, 0);
        size_t runs = 0;
        for(uint64_t ts=kFrame; ts<1900*kMs; ts+=kFrame){
            const Decision d = decideAndRun(s, quiet(ts));
            CHECK(d.action == Action::IDLE);
            runs += d.action == Action::RUN_DETECTOR;
        }
        CHECK(runs == 0);
        CHECK(s.state() == RuntimeState::IDLE);

        // Idle safety poll after idle_poll_ms.
        CHECK(is(decideAndRun(s, quiet(2000*kMs)), Action::RUN_DETECTOR, "idle_poll"));
    }

    // Motion + no track -> RUN_DETECTOR, state moves to RECHECK then ACTIVE on a person.
    {
        AdaptiveScheduler s(adaptiveConfig());
        decideAndRun(s, quiet(0));
        s.onDetectorResult(0, 0);

        SchedulerInputs in = quiet(500*kMs);
        in.motion_active = true;
        in.motion_event = true;
        const Decision d = decideAndRun(s, in);
        CHECK(is(d, Action::RUN_DETECTOR, "motion_event"));
        CHECK(s.state() == RuntimeState::RECHECK);

        s.onDetectorResult(1, 500*kMs);
        CHECK(s.state() == RuntimeState::ACTIVE);
    }

    // Recheck finding nobody returns to IDLE.
    {
        AdaptiveScheduler s(adaptiveConfig());
        SchedulerInputs in = quiet(0);
        in.motion_active = true;
        in.motion_event = true;
        decideAndRun(s, in);
        CHECK(s.state() == RuntimeState::RECHECK);
        s.onDetectorResult(0, 0);
        CHECK(s.state() == RuntimeState::IDLE);
    }

    // Continuous motion without a track rechecks at min_semantic_interval, not every frame.
    {
        AdaptiveScheduler s(adaptiveConfig());
        size_t runs = 0;
        for(uint64_t ts=0; ts<1000*kMs; ts+=kFrame){
            SchedulerInputs in = quiet(ts);
            in.motion_active = true;
            runs += decideAndRun(s, in).action == Action::RUN_DETECTOR;
        }
        CHECK(runs >= 4 && runs <= 6);   // 1 s / 200 ms
    }

    // Valid track + quiet scene -> TRACK_ONLY until max semantic age.
    {
        AdaptiveScheduler s(adaptiveConfig());
        s.markDetectorRun(0);
        s.onDetectorResult(1, 0);
        size_t runs = 0;
        for(uint64_t ts=kFrame; ts<990*kMs; ts+=kFrame){
            SchedulerInputs in = quiet(ts);
            in.confirmed_tracks = 1;
            const Decision d = decideAndRun(s, in);
            CHECK(d.action == Action::TRACK_ONLY);
            runs += d.action == Action::RUN_DETECTOR;
        }
        CHECK(runs == 0);
        CHECK(s.state() == RuntimeState::ACTIVE);

        // Maximum semantic age exceeded -> RUN_DETECTOR.
        SchedulerInputs in = quiet(1000*kMs);
        in.confirmed_tracks = 1;
        CHECK(is(decideAndRun(s, in), Action::RUN_DETECTOR, "max_semantic_age"));
    }

    // Tracker failure -> RUN_DETECTOR immediately.
    {
        AdaptiveScheduler s(adaptiveConfig());
        s.markDetectorRun(0);
        SchedulerInputs in = quiet(50*kMs);
        in.confirmed_tracks = 1;
        in.tracker_failed = true;
        CHECK(is(decideAndRun(s, in), Action::RUN_DETECTOR, "tracker_failed"));
    }

    // Large unexplained motion with tracks -> RUN_DETECTOR; explained motion -> TRACK_ONLY.
    {
        AdaptiveScheduler s(adaptiveConfig());
        s.markDetectorRun(0);

        SchedulerInputs explained = quiet(300*kMs);
        explained.confirmed_tracks = 1;
        explained.motion_active = true;
        explained.unexplained_motion_fraction = 0.05f;
        CHECK(is(decideAndRun(s, explained), Action::TRACK_ONLY, "track"));

        SchedulerInputs unexplained = explained;
        unexplained.ts_ns = 340*kMs;
        unexplained.unexplained_motion_fraction = 0.8f;
        CHECK(is(decideAndRun(s, unexplained), Action::RUN_DETECTOR, "unexplained_motion"));
    }

    // Abrupt motion shortens the semantic interval.
    {
        AdaptiveScheduler s(adaptiveConfig());
        s.markDetectorRun(0);
        SchedulerInputs in = quiet(100*kMs);
        in.confirmed_tracks = 1;
        in.abrupt_motion = true;
        CHECK(is(decideAndRun(s, in), Action::TRACK_ONLY, "track"));
        in.ts_ns = 210*kMs;
        CHECK(is(decideAndRun(s, in), Action::RUN_DETECTOR, "abrupt_motion"));
    }

    // Busy detector: periodic checks are dropped, urgent ones still request.
    {
        AdaptiveScheduler s(adaptiveConfig());
        s.markDetectorRun(0);
        SchedulerInputs in = quiet(1500*kMs);
        in.confirmed_tracks = 1;
        in.detector_busy = true;
        CHECK(is(decideAndRun(s, in), Action::TRACK_ONLY, "detector_busy"));

        in.tracker_failed = true;
        CHECK(is(decideAndRun(s, in), Action::RUN_DETECTOR, "tracker_failed"));
    }

    // Detector gaps are variable rather than fixed.
    {
        AdaptiveScheduler s(adaptiveConfig());
        uint64_t last_run = 0;
        bool have_last = false;
        uint64_t min_gap = UINT64_MAX, max_gap = 0;
        for(uint64_t ts=0; ts<6000*kMs; ts+=kFrame){
            SchedulerInputs in = quiet(ts);
            const bool person = ts >= 1000*kMs && ts < 4000*kMs;
            in.confirmed_tracks = person ? 1 : 0;
            in.motion_active = (ts >= 1000*kMs && ts < 1100*kMs) || (ts >= 2500*kMs && ts < 2600*kMs);
            in.unexplained_motion_fraction = in.motion_active ? 0.9f : 0.0f;
            const Decision d = decideAndRun(s, in);
            if(d.action == Action::RUN_DETECTOR){
                s.onDetectorResult(person ? 1 : 0, ts);
                if(have_last){
                    min_gap = std::min(min_gap, ts - last_run);
                    max_gap = std::max(max_gap, ts - last_run);
                }
                last_run = ts;
                have_last = true;
            }
        }
        CHECK(max_gap > 2*min_gap);
    }

    // Startup frames always run.
    {
        SchedulerConfig c = adaptiveConfig();
        c.startup_frames = 3;
        AdaptiveScheduler s(c);
        for(int i=0; i<3; i++) CHECK(is(decideAndRun(s, quiet(i*kFrame)), Action::RUN_DETECTOR, "startup"));
        CHECK(decideAndRun(s, quiet(3*kFrame)).action == Action::IDLE);
    }

    // Mode: always.
    {
        SchedulerConfig c;
        c.mode = SchedulerMode::ALWAYS;
        AdaptiveScheduler s(c);
        for(int i=0; i<20; i++) CHECK(is(decideAndRun(s, quiet(i*kFrame)), Action::RUN_DETECTOR, "always"));
    }

    // Mode: fixed interval (R4).
    {
        SchedulerConfig c;
        c.mode = SchedulerMode::FIXED_INTERVAL;
        c.startup_frames = 0;
        c.fixed_interval_ms = 500;
        AdaptiveScheduler s(c);
        size_t runs = 0;
        for(uint64_t ts=0; ts<3000*kMs; ts+=kFrame) runs += decideAndRun(s, quiet(ts)).action == Action::RUN_DETECTOR;
        CHECK(runs == 6);
    }

    // Mode: motion gated (R5): fixed interval while active, idle polls when quiet.
    {
        SchedulerConfig c;
        c.mode = SchedulerMode::MOTION_GATED;
        c.startup_frames = 0;
        c.fixed_interval_ms = 200;
        c.active_hold_ms = 0;
        AdaptiveScheduler s(c);
        size_t quiet_runs = 0, active_runs = 0;
        for(uint64_t ts=0; ts<2000*kMs; ts+=kFrame){
            quiet_runs += decideAndRun(s, quiet(ts)).action == Action::RUN_DETECTOR;
        }
        for(uint64_t ts=2000*kMs; ts<3000*kMs; ts+=kFrame){
            SchedulerInputs in = quiet(ts);
            in.motion_active = true;
            active_runs += decideAndRun(s, in).action == Action::RUN_DETECTOR;
        }
        CHECK(quiet_runs == 1);
        CHECK(active_runs >= 4);
    }

    // Mode parsing.
    CHECK(schedulerModeFromString("adaptive") == SchedulerMode::ADAPTIVE);
    CHECK(schedulerModeFromString("motion_gated") == SchedulerMode::MOTION_GATED);
    bool threw = false;
    try{ schedulerModeFromString("every_frame"); } catch(const std::invalid_argument&){ threw = true; }
    CHECK(threw);

    std::cout << "All scheduler tests passed.\n";
    return 0;
}
