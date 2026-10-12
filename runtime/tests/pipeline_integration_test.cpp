#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include <edge_ai/capture/replay_source.hpp>
#include <edge_ai/detector/simulated_detector.hpp>
#include <edge_ai/metrics/profiler.hpp>
#include <edge_ai/pipeline/pipeline.hpp>

#include "test_common.hpp"

namespace {

constexpr size_t W = 320;
constexpr size_t H = 240;
constexpr uint64_t kFrame = 33*kMs;

// Synthetic scene: empty, then a "person" block walks in, stops, and leaves.
struct ScriptedScene{
    static constexpr uint64_t enter_frame = 60;
    static constexpr uint64_t stop_frame = 120;
    static constexpr uint64_t leave_frame = 200;
    static constexpr uint64_t total = 280;

    static float personX(uint64_t frame){
        if(frame < enter_frame || frame >= leave_frame) return -1.0f;
        const uint64_t walked = std::min(frame, stop_frame) - enter_frame;
        return 20.0f + 2.0f*static_cast<float>(walked);
    }

    static Detection personBox(uint64_t frame){
        const float x = personX(frame);
        return {x, 60.0f, x + 40.0f, 180.0f, 0.9f};
    }
};

// Detector oracle: returns the person box for the frame that was actually submitted.
std::vector<Detection> oracle(uint64_t ts_ns){
    const uint64_t frame = ts_ns / kFrame;
    if(ScriptedScene::personX(frame) < 0.0f) return {};
    return {ScriptedScene::personBox(frame)};
}

struct RunResult{
    size_t frames = 0;
    size_t calls = 0;
    size_t frames_with_person = 0;
    size_t frames_person_tracked = 0;
    float max_center_error = 0.0f;
    size_t tracks_after_leave = 0;
};

int runScripted(SchedulerMode mode, bool async, uint32_t delay_ms, RunResult& out){
    RuntimeConfig config;
    config.scheduler.mode = mode;
    config.pipeline.async = async;
    config.model.input_width = 64;
    config.model.input_height = 48;

    SimulatedDetector detector(64, 48, oracle, delay_ms);
    Profiler profiler;
    Pipeline pipeline(config, detector, profiler);

    for(uint64_t f=0; f<ScriptedScene::total; f++){
        const float x = ScriptedScene::personX(f);
        TestImage image = x < 0.0f
            ? makeSceneRGB(W, H, 70)
            : makeSceneRGB(W, H, 70, static_cast<int>(x), 60, static_cast<int>(x) + 40, 180, 220);

        const uint64_t ts = f*kFrame;
        pipeline.step(makeFrame(image.view, ts, f));

        // Real-time pacing so a delayed detector really is late relative to frames.
        if(async && delay_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(33));

        const bool person = x >= 0.0f;
        out.frames_with_person += person;

        const auto boxes = pipeline.tracker().boxes(true);
        if(person && !boxes.empty() && f > ScriptedScene::enter_frame + 10){
            out.frames_person_tracked++;
            const Detection truth = ScriptedScene::personBox(f);
            const float err = std::fabs((boxes[0].x1 + boxes[0].x2)*0.5f - (truth.x1 + truth.x2)*0.5f);
            out.max_center_error = std::max(out.max_center_error, err);
        }
    }

    // Let the person-leaving refresh settle, then confirm the track is gone.
    for(uint64_t f=ScriptedScene::total; f<ScriptedScene::total + 60; f++){
        TestImage image = makeSceneRGB(W, H, 70);
        pipeline.step(makeFrame(image.view, f*kFrame, f));
        if(async && delay_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }
    pipeline.drain();

    out.frames = profiler.frames();
    out.calls = profiler.detectorCalls();
    out.tracks_after_leave = pipeline.tracker().confirmedCount();
    return 0;
}

} // namespace

int main(int argc, char** argv){
    // Every-frame baseline: one detector call per frame.
    {
        RunResult r;
        if(runScripted(SchedulerMode::ALWAYS, false, 0, r)) return 1;
        CHECK(r.calls == r.frames);
    }

    // Adaptive pipeline: far fewer calls, person tracked, track cleared after leaving.
    {
        RunResult r;
        if(runScripted(SchedulerMode::ADAPTIVE, false, 0, r)) return 1;
        std::cout << "adaptive sync: frames " << r.frames << ", calls " << r.calls
                  << ", tracked " << r.frames_person_tracked << "/" << r.frames_with_person
                  << ", max center error " << r.max_center_error << " px\n";
        CHECK(r.calls < r.frames/3);
        CHECK(r.frames_person_tracked > (r.frames_with_person*8)/10);
        CHECK(r.max_center_error < 25.0f);
        CHECK(r.tracks_after_leave == 0);
    }

    // Async with injected latency: delayed results do not corrupt the present-time track.
    for(uint32_t delay : {100u, 300u}){
        RunResult r;
        if(runScripted(SchedulerMode::ADAPTIVE, true, delay, r)) return 1;
        std::cout << "adaptive async " << delay << " ms: frames " << r.frames << ", calls " << r.calls
                  << ", tracked " << r.frames_person_tracked << "/" << r.frames_with_person
                  << ", max center error " << r.max_center_error << " px\n";
        CHECK(r.calls < r.frames/3);
        CHECK(r.frames_person_tracked > (r.frames_with_person*6)/10);
        CHECK(r.max_center_error < 40.0f);
        CHECK(r.tracks_after_leave == 0);
    }

    // Real recording through ReplaySource with the full pipeline and metrics outputs.
    if(argc >= 2){
        const std::string video = argv[1];
        const std::filesystem::path out_dir = std::filesystem::temp_directory_path() / "edge_ai_pipeline_test";
        std::filesystem::create_directories(out_dir);

        RuntimeConfig config;
        config.scheduler.mode = SchedulerMode::ADAPTIVE;
        SimulatedDetector detector(320, 256, [](uint64_t){ return std::vector<Detection>{{100, 100, 300, 600, 0.8f}}; }, 20);

        Profiler profiler;
        profiler.openOutputs(out_dir.string(), true, true);
        Pipeline pipeline(config, detector, profiler);

        // Loop the clip: bounded memory and correct handling of time restarting.
        double first_rss = 0.0;
        for(int loop=0; loop<5; loop++){
            ReplaySource source(video, ReplayMode::UNLIMITED);
            const size_t n = pipeline.run(source);
            CHECK(n > 0);
            if(loop == 0) first_rss = sampleResourceUsage().peak_rss_mb;
        }
        const double final_rss = sampleResourceUsage().peak_rss_mb;
        std::cout << "replay x5: frames " << profiler.frames() << ", calls " << profiler.detectorCalls()
                  << ", peak RSS " << first_rss << " -> " << final_rss << " MB\n";

        CHECK(profiler.detectorCalls() > 0);
        CHECK(profiler.detectorCalls() < profiler.frames());
        CHECK(final_rss < first_rss + 64.0);
        CHECK(pipeline.tracker().historySize() < 200);

        profiler.writeSummary((out_dir / "summary.json").string());
        CHECK(std::filesystem::file_size(out_dir / "summary.json") > 0);
        CHECK(std::filesystem::file_size(out_dir / "frames.csv") > 0);
        CHECK(std::filesystem::file_size(out_dir / "detections.jsonl") > 0);
    }

    std::cout << "All pipeline integration tests passed.\n";
    return 0;
}
