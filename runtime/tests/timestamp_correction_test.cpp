#include <cmath>
#include <iostream>

#include <edge_ai/tracker/tracker.hpp>

#include "test_common.hpp"

namespace {

constexpr float kSpeed = 200.0f;     // px/s, person walking right
constexpr uint64_t kFrame = 33*kMs;

float truthX(uint64_t ts_ns){
    return 100.0f + kSpeed*static_cast<float>(ts_ns/1e9);
}

Detection truthBox(uint64_t ts_ns){
    return makeBox(truthX(ts_ns), 240, 40, 80);
}

// Builds a track with a learned velocity using on-time detections up to start_ns.
void warmTracker(Tracker& tracker, uint64_t start_ns){
    for(uint64_t ts=0; ts<=start_ns; ts+=kFrame){
        tracker.predict(ts);
        tracker.update(makeBatch(ts, {truthBox(ts)}));
    }
}

} // namespace

int main(){
    // Delayed result for T arrives at T + delay. The correction must land on the state at T
    // and be propagated forward, not be applied as if the box were observed "now".
    for(uint64_t delay_ms : {100ULL, 200ULL, 300ULL, 500ULL}){
        TrackerConfig config;
        Tracker tracker(config);

        const uint64_t t_obs = 1000*kMs;
        warmTracker(tracker, t_obs);

        // Frames keep arriving while inference runs.
        uint64_t now = t_obs;
        while(now + kFrame <= t_obs + delay_ms*kMs){
            now += kFrame;
            tracker.predict(now);
        }

        tracker.update(makeBatch(t_obs, {truthBox(t_obs)}));
        CHECK(tracker.tracks().size() == 1);
        CHECK(tracker.currentTs() == now);

        const Track& t = tracker.tracks()[0];
        const float error = std::fabs(t.cx - truthX(now));

        // Naive approach: treating the old box as current pulls the track back by speed * delay.
        const float naive_error = kSpeed*static_cast<float>((now - t_obs)/1e9);

        std::cout << "delay " << delay_ms << " ms: timestamp-aware error " << error
                  << " px, naive error " << naive_error << " px\n";

        CHECK(error < 3.0f);
        CHECK(error < naive_error*0.25f);
        CHECK(t.last_detection_ts_ns == t_obs);
        CHECK(tracker.staleBatches() == 0);
        CHECK(tracker.tracks()[0].id == 1);
    }

    // History is rewritten: a later delayed result builds on the corrected past.
    {
        Tracker tracker;
        warmTracker(tracker, 500*kMs);
        for(uint64_t ts = 500*kMs + kFrame; ts <= 800*kMs; ts += kFrame) tracker.predict(ts);
        tracker.update(makeBatch(500*kMs + kFrame*3, {truthBox(500*kMs + kFrame*3)}));
        const size_t history = tracker.historySize();
        CHECK(history > 0);
        CHECK(std::fabs(tracker.tracks()[0].cx - truthX(tracker.currentTs())) < 3.0f);
    }

    // A person who appears in a delayed frame gets a track propagated to the present.
    {
        Tracker tracker;
        for(uint64_t ts=0; ts<=400*kMs; ts+=kFrame) tracker.predict(ts);
        tracker.update(makeBatch(200*kMs, {makeBox(300, 240, 40, 80)}));
        CHECK(tracker.tracks().size() == 1);
        CHECK(tracker.currentTs() == 396*kMs);
        CHECK(tracker.tracks()[0].last_detection_ts_ns == 200*kMs);
    }

    // A result older than the history window falls back to latest-result correction.
    {
        TrackerConfig config;
        config.history_ms = 200;
        Tracker tracker(config);
        warmTracker(tracker, 300*kMs);
        for(uint64_t ts = 300*kMs + kFrame; ts <= 1200*kMs; ts += kFrame) tracker.predict(ts);
        tracker.update(makeBatch(100*kMs, {truthBox(100*kMs)}));
        CHECK(tracker.staleBatches() == 1);
        CHECK(tracker.tracks().size() >= 1);
    }

    std::cout << "All timestamp correction tests passed.\n";
    return 0;
}
