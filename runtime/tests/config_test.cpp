#include <iostream>
#include <stdexcept>
#include <string>

#include <edge_ai/config/runtime_config.hpp>

#include "test_common.hpp"

namespace {

bool rejects(const std::string& json){
    try{
        parseConfig(json);
    }
    catch(const std::invalid_argument&){
        return true;
    }
    return false;
}

} // namespace

int main(){
    // Defaults mirror the Python adaptive configuration.
    {
        const RuntimeConfig c = parseConfig("{}");
        CHECK_NEAR(c.model.confidence, 0.25f, 1e-6);
        CHECK_NEAR(c.model.iou, 0.45f, 1e-6);
        CHECK(c.model.input_width == 320 && c.model.input_height == 256);
        CHECK(c.motion.width == 160 && c.motion.height == 120);
        CHECK(c.motion.pixel_threshold == 25);
        CHECK(c.motion.minimum_changed_area == 192);
        CHECK(c.motion.cooldown_ms == 250);
        CHECK(c.scheduler.mode == SchedulerMode::ADAPTIVE);
        CHECK(c.pipeline.async);
        CHECK(c.pipeline.busy_policy == BusyPolicy::LATEST);
    }

    // Overrides.
    {
        const RuntimeConfig c = parseConfig(R"({
            "model": {"path": "m.onnx", "confidence": 0.4, "threads": 2},
            "motion": {"pixel_threshold": 40, "minimum_changed_area": 300},
            "tracker": {"max_misses": 5},
            "scheduler": {"mode": "fixed_interval", "fixed_interval_ms": 250},
            "pipeline": {"async": false, "busy_policy": "skip", "detector": "simulated", "max_frames": 10},
            "camera": {"width": 320, "height": 240},
            "output": {"directory": "runs/x", "overlay": false}
        })");
        CHECK(c.model.path == "m.onnx");
        CHECK_NEAR(c.model.confidence, 0.4f, 1e-6);
        CHECK(c.model.threads == 2);
        CHECK(c.motion.pixel_threshold == 40);
        CHECK(c.motion.minimum_changed_area == 300);
        CHECK(c.tracker.max_misses == 5);
        CHECK(c.scheduler.mode == SchedulerMode::FIXED_INTERVAL);
        CHECK(c.scheduler.fixed_interval_ms == 250);
        CHECK(!c.pipeline.async);
        CHECK(c.pipeline.busy_policy == BusyPolicy::SKIP);
        CHECK(c.pipeline.detector == "simulated");
        CHECK(c.pipeline.max_frames == 10);
        CHECK(c.camera.width == 320);
        CHECK(c.output.directory == "runs/x");
        CHECK(!c.output.overlay);

        // Round trip through the effective-config dump.
        const RuntimeConfig again = parseConfig(configToJson(c));
        CHECK(again.scheduler.mode == SchedulerMode::FIXED_INTERVAL);
        CHECK(again.motion.pixel_threshold == 40);
        CHECK(again.pipeline.busy_policy == BusyPolicy::SKIP);
    }

    // Invalid values and typos are rejected rather than silently defaulted.
    CHECK(rejects("not json"));
    CHECK(rejects(R"({"motion": {"pixel_treshold": 30}})"));
    CHECK(rejects(R"({"unknown_section": {}})"));
    CHECK(rejects(R"({"model": {"confidence": 1.5}})"));
    CHECK(rejects(R"({"model": {"confidence": "high"}})"));
    CHECK(rejects(R"({"motion": {"pixel_threshold": 0}})"));
    CHECK(rejects(R"({"motion": {"minimum_changed_area": 999999}})"));
    CHECK(rejects(R"({"scheduler": {"mode": "every_frame"}})"));
    CHECK(rejects(R"({"scheduler": {"min_semantic_interval_ms": 2000, "max_semantic_age_ms": 1000}})"));
    CHECK(rejects(R"({"pipeline": {"busy_policy": "queue"}})"));
    CHECK(rejects(R"({"pipeline": {"detector": "tensorrt"}})"));
    CHECK(rejects(R"({"camera": {"width": 641}})"));

    // Shipped configuration files parse.
    for(const char* path : {EDGE_AI_CONFIG_DIR "/runtime.json",
                            EDGE_AI_CONFIG_DIR "/ladder/r1.json",
                            EDGE_AI_CONFIG_DIR "/ladder/r2.json",
                            EDGE_AI_CONFIG_DIR "/ladder/r3.json",
                            EDGE_AI_CONFIG_DIR "/ladder/r4.json",
                            EDGE_AI_CONFIG_DIR "/ladder/r5.json",
                            EDGE_AI_CONFIG_DIR "/ladder/r6.json"}){
        try{
            loadConfig(path);
        }
        catch(const std::exception& error){
            std::cerr << path << ": " << error.what() << '\n';
            return 1;
        }
    }

    std::cout << "All config tests passed.\n";
    return 0;
}
