#include <cmath>
#include <iostream>

#include <edge_ai/tracker/tracker.hpp>

#include "test_common.hpp"

int main(){
    TrackerConfig config;   // min_hits 2, confirm_score .6, max_misses 3, max age 2 s

    // First confident detection creates a confirmed track.
    {
        Tracker tracker(config);
        tracker.predict(0);
        tracker.update(makeBatch(0, {makeBox(100, 100, 40, 80, 0.9f)}));
        CHECK(tracker.tracks().size() == 1);
        CHECK(tracker.confirmedCount() == 1);
        CHECK(tracker.tracks()[0].id == 1);
        CHECK(tracker.tracks()[0].hits == 1);
    }

    // A low-confidence detection stays tentative until a second hit.
    {
        Tracker tracker(config);
        tracker.predict(0);
        tracker.update(makeBatch(0, {makeBox(100, 100, 40, 80, 0.4f)}));
        CHECK(tracker.tracks().size() == 1);
        CHECK(tracker.confirmedCount() == 0);
        tracker.predict(100*kMs);
        tracker.update(makeBatch(100*kMs, {makeBox(102, 100, 40, 80, 0.4f)}));
        CHECK(tracker.confirmedCount() == 1);
        CHECK(tracker.tracks()[0].hits == 2);
    }

    // Repeated detections keep the ID; prediction continues between detector calls.
    {
        Tracker tracker(config);
        // Person walks right at 100 px/s, detector every 200 ms.
        for(int k=0; k<=5; k++){
            const uint64_t ts = static_cast<uint64_t>(k)*200*kMs;
            tracker.predict(ts);
            tracker.update(makeBatch(ts, {makeBox(100.0f + 20.0f*k, 200, 40, 80)}));
        }
        CHECK(tracker.tracks().size() == 1);
        const int id = tracker.tracks()[0].id;
        CHECK(id == 1);
        CHECK(tracker.tracks()[0].vx > 50.0f);   // learned rightward velocity

        // No detector for 300 ms: the track keeps moving.
        const float before = tracker.tracks()[0].cx;
        tracker.predict(1300*kMs);
        CHECK(tracker.tracks().size() == 1);
        CHECK(tracker.tracks()[0].cx > before + 15.0f);
        CHECK(tracker.tracks()[0].id == id);
    }

    // Missed detections increment misses; the track is deleted after max_misses.
    {
        Tracker tracker(config);
        tracker.predict(0);
        tracker.update(makeBatch(0, {makeBox(100, 100, 40, 80)}));
        CHECK(tracker.weak());   // young: velocity not yet known

        tracker.predict(100*kMs);
        tracker.update(makeBatch(100*kMs, {}));
        CHECK(tracker.tracks().size() == 1);
        CHECK(tracker.tracks()[0].misses == 1);
        CHECK(tracker.weak());

        tracker.predict(200*kMs);
        tracker.update(makeBatch(200*kMs, {}));
        CHECK(tracker.tracks().size() == 1);
        CHECK(!tracker.failed());

        tracker.predict(300*kMs);
        tracker.update(makeBatch(300*kMs, {}));
        CHECK(tracker.tracks().empty());
        CHECK(tracker.failed());   // a confirmed person was lost
        tracker.acknowledgeFailure();
        CHECK(!tracker.failed());
    }

    // Stale tracks without detector support age out.
    {
        Tracker tracker(config);
        tracker.predict(0);
        tracker.update(makeBatch(0, {makeBox(100, 100, 40, 80)}));
        tracker.predict(1900*kMs);
        CHECK(tracker.tracks().size() == 1);
        tracker.predict(2100*kMs);
        CHECK(tracker.tracks().empty());
    }

    // A track predicted out of the frame is removed.
    {
        Tracker tracker(config);
        tracker.setFrameSize(640, 480);
        for(int k=0; k<3; k++){
            const uint64_t ts = static_cast<uint64_t>(k)*100*kMs;
            tracker.predict(ts);
            tracker.update(makeBatch(ts, {makeBox(560.0f + 30.0f*k, 200, 40, 80)}));
        }
        tracker.predict(1500*kMs);
        CHECK(tracker.tracks().empty());
    }

    // Detector refresh corrects drift: a wrong velocity is pulled back by detections.
    {
        Tracker tracker(config);
        tracker.predict(0);
        tracker.update(makeBatch(0, {makeBox(300, 200, 40, 80)}));
        tracker.predict(100*kMs);
        tracker.update(makeBatch(100*kMs, {makeBox(330, 200, 40, 80)}));   // fast step: high vx

        // Person actually stopped at 330. Predictions drift, detections pull back.
        for(int k=2; k<=10; k++){
            const uint64_t ts = static_cast<uint64_t>(k)*100*kMs;
            tracker.predict(ts);
            tracker.update(makeBatch(ts, {makeBox(330, 200, 40, 80)}));
        }
        CHECK(tracker.tracks().size() == 1);
        CHECK_NEAR(tracker.tracks()[0].cx, 330.0f, 3.0f);
        CHECK(std::fabs(tracker.tracks()[0].vx) < 30.0f);
    }

    // Two people get separate, stable IDs.
    {
        Tracker tracker(config);
        for(int k=0; k<4; k++){
            const uint64_t ts = static_cast<uint64_t>(k)*100*kMs;
            tracker.predict(ts);
            tracker.update(makeBatch(ts, {makeBox(100.0f + 5*k, 200, 40, 80), makeBox(400.0f - 5*k, 200, 40, 80)}));
        }
        CHECK(tracker.tracks().size() == 2);
        CHECK(tracker.tracks()[0].id == 1);
        CHECK(tracker.tracks()[1].id == 2);
        CHECK(tracker.tracks()[0].cx < tracker.tracks()[1].cx);
    }

    // Abrupt direction change on a confirmed track is reported.
    {
        Tracker tracker(config);
        for(int k=0; k<=4; k++){
            const uint64_t ts = static_cast<uint64_t>(k)*100*kMs;
            tracker.predict(ts);
            tracker.update(makeBatch(ts, {makeBox(200.0f + 10.0f*k, 200, 40, 80)}));
            CHECK(!tracker.abruptMotion());
        }
        tracker.predict(500*kMs);
        tracker.update(makeBatch(500*kMs, {makeBox(220, 200, 40, 80)}));   // reverses: 30 px behind prediction
        CHECK(tracker.abruptMotion());
    }

    // Fast mover with little box overlap still associates via the distance fallback.
    {
        Tracker tracker(config);
        tracker.predict(0);
        tracker.update(makeBatch(0, {makeBox(100, 200, 40, 80)}));
        tracker.predict(100*kMs);
        tracker.update(makeBatch(100*kMs, {makeBox(130, 200, 40, 80)}));   // IoU 0.14, distance 0.375 h
        CHECK(tracker.tracks().size() == 1);
        CHECK(tracker.tracks()[0].hits == 2);
    }

    // A still scene holds the track in place and bleeds off velocity.
    {
        Tracker tracker(config);
        for(int k=0; k<=3; k++){
            const uint64_t ts = static_cast<uint64_t>(k)*100*kMs;
            tracker.predict(ts);
            tracker.update(makeBatch(ts, {makeBox(100.0f + 10.0f*k, 200, 40, 80)}));
        }
        const float stopped_at = tracker.tracks()[0].cx;
        for(int k=4; k<=12; k++){
            tracker.predict(static_cast<uint64_t>(k)*100*kMs, [](const Track&){ return true; });
        }
        CHECK_NEAR(tracker.tracks()[0].cx, stopped_at, 1e-4);
        CHECK(std::fabs(tracker.tracks()[0].vx) < 1.0f);
    }

    std::cout << "All tracker tests passed.\n";
    return 0;
}
