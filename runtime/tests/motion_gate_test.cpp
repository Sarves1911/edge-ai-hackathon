#include <cstdint>
#include <iostream>
#include <random>

#include <edge_ai/motion/motion_gate.hpp>

#include "test_common.hpp"

namespace {

constexpr size_t W = 320;
constexpr size_t H = 240;
constexpr uint64_t kFrame = 33*kMs;

// Runs the warm-up period (reference frame + warmup_frames) on a static scene.
int warmUp(MotionGate& gate, const TestImage& scene, uint64_t& ts){
    for(int i=0; i<4; i++){
        const MotionResult r = gate.update(scene.view, ts);
        CHECK(!r.active && !r.event);
        ts += kFrame;
    }
    return 0;
}

} // namespace

int main(){
    MotionConfig config;   // 160x120, threshold 25, area 192, cooldown 250 ms, warmup 3
    const TestImage empty = makeSceneRGB(W, H, 80);

    // Identical frames -> zero motion, no trigger.
    {
        MotionGate gate(config);
        uint64_t ts = 0;
        if(warmUp(gate, empty, ts)) return 1;
        for(int i=0; i<30; i++){
            const MotionResult r = gate.update(empty.view, ts);
            CHECK(r.changed_pixels == 0);
            CHECK(r.score == 0.0f);
            CHECK(!r.active && !r.event);
            ts += kFrame;
        }
    }

    // Sensor-like noise below the pixel threshold never triggers.
    {
        MotionGate gate(config);
        uint64_t ts = 0;
        if(warmUp(gate, empty, ts)) return 1;

        std::mt19937 rng(7);
        std::uniform_int_distribution<int> noise(-10, 10);
        for(int i=0; i<30; i++){
            TestImage noisy = makeImage(W, H, PixelFormat::RGB24, 0, [&](size_t, size_t, uint8_t* p){
                const uint8_t v = static_cast<uint8_t>(80 + noise(rng));
                p[0] = v; p[1] = v; p[2] = v;
            });
            const MotionResult r = gate.update(noisy.view, ts);
            CHECK(!r.active && !r.event);
            ts += kFrame;
        }
    }

    // A small changed region (below minimum_changed_area) is measured but not active.
    {
        MotionGate gate(config);
        uint64_t ts = 0;
        if(warmUp(gate, empty, ts)) return 1;
        const TestImage speck = makeSceneRGB(W, H, 80, 100, 100, 110, 110);   // 5x5 at motion scale
        const MotionResult r = gate.update(speck.view, ts);
        CHECK(r.changed_pixels > 0);
        CHECK(r.changed_pixels < config.minimum_changed_area);
        CHECK(!r.active && !r.event);
    }

    // Person-like region entering after warm-up triggers immediately.
    {
        MotionGate gate(config);
        uint64_t ts = 0;
        if(warmUp(gate, empty, ts)) return 1;

        const TestImage entered = makeSceneRGB(W, H, 80, 20, 40, 100, 200);   // 40x80 at motion scale
        const MotionResult r = gate.update(entered.view, ts);
        CHECK(r.active);
        CHECK(r.event);
        CHECK(r.changed_pixels == 40*80);
        CHECK_NEAR(r.score, 3200.0/19200.0, 1e-6);

        // Box around the entered region explains the motion; no boxes leaves it unexplained.
        CHECK_NEAR(gate.changedFractionOutside({{20, 40, 100, 200, 0.9f}}), 0.0f, 1e-6);
        CHECK_NEAR(gate.changedFractionOutside({}), 1.0f, 1e-6);
        CHECK_NEAR(gate.changedFractionOutside({{200, 40, 280, 200, 0.9f}}), 1.0f, 1e-6);
        ts += kFrame;

        // Moving within cooldown: still active, but no new event.
        const TestImage moved = makeSceneRGB(W, H, 80, 60, 40, 140, 200);
        const MotionResult m = gate.update(moved.view, ts);
        CHECK(m.active);
        CHECK(!m.event);

        // After the cooldown a new event is allowed.
        ts += 300*kMs;
        const MotionResult n = gate.update(entered.view, ts);
        CHECK(n.active && n.event);
    }

    // Startup transients inside the warm-up never trigger.
    {
        MotionGate gate(config);
        const TestImage bright = makeSceneRGB(W, H, 220);
        CHECK(!gate.update(empty.view, 0).active);
        const MotionResult r = gate.update(bright.view, kFrame);
        CHECK(r.changed_pixels == 160*120);
        CHECK(!r.active && !r.event);
    }

    // Time moving backwards (looped replay) resets the reference frame.
    {
        MotionGate gate(config);
        uint64_t ts = 10*kFrame;
        if(warmUp(gate, empty, ts)) return 1;
        const TestImage entered = makeSceneRGB(W, H, 80, 20, 40, 100, 200);
        const MotionResult r = gate.update(entered.view, 0);
        CHECK(r.changed_pixels == 0);
        CHECK(!r.active);
    }

    // YUYV luma and GRAY8 are accepted directly.
    {
        MotionGate gate(config);
        const TestImage gray = makeImage(W, H, PixelFormat::GRAY8, 3, [](size_t, size_t, uint8_t* p){ *p = 90; });
        const TestImage yuyv = makeImage(W, H, PixelFormat::YUYV, 0, [](size_t, size_t, uint8_t* p){ p[0] = 90; p[1] = 128; });
        CHECK(gate.update(gray.view, 0).changed_pixels == 0);
        CHECK(gate.update(yuyv.view, kFrame).changed_pixels == 0);
    }

    std::cout << "All motion gate tests passed.\n";
    return 0;
}
