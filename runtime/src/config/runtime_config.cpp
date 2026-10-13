#include <edge_ai/config/runtime_config.hpp>

#include <fstream>
#include <initializer_list>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

using nlohmann::json;

namespace {

void checkKeys(const json& obj, const std::string& section, std::initializer_list<const char*> allowed){
    if(!obj.is_object()) throw std::invalid_argument("config section '" + section + "' must be an object");

    for(auto it = obj.begin(); it != obj.end(); ++it){
        bool known = false;
        for(const char* key : allowed) if(it.key() == key) known = true;
        if(!known) throw std::invalid_argument("unknown config key '" + section + "." + it.key() + "'");
    }
}

template <typename T>
void read(const json& obj, const char* key, T& out){
    if(!obj.contains(key)) return;
    try{
        out = obj.at(key).get<T>();
    }
    catch(const json::exception& error){
        throw std::invalid_argument(std::string("invalid config value for '") + key + "': " + error.what());
    }
}

const json& section(const json& root, const char* name){
    static const json empty = json::object();
    return root.contains(name) ? root.at(name) : empty;
}

} // namespace

RuntimeConfig parseConfig(const std::string& json_text){
    json root;
    try{
        root = json::parse(json_text);
    }
    catch(const json::exception& error){
        throw std::invalid_argument(std::string("invalid config JSON: ") + error.what());
    }

    checkKeys(root, "root", {"model", "motion", "tracker", "scheduler", "pipeline", "camera", "output"});

    RuntimeConfig c;

    const json& model = section(root, "model");
    checkKeys(model, "model", {"path", "confidence", "iou", "class_id", "threads", "input_width", "input_height"});
    read(model, "path", c.model.path);
    read(model, "confidence", c.model.confidence);
    read(model, "iou", c.model.iou);
    read(model, "class_id", c.model.class_id);
    read(model, "threads", c.model.threads);
    read(model, "input_width", c.model.input_width);
    read(model, "input_height", c.model.input_height);

    const json& motion = section(root, "motion");
    checkKeys(motion, "motion", {"width", "height", "pixel_threshold", "minimum_changed_area", "cooldown_ms", "warmup_frames", "box_margin"});
    read(motion, "width", c.motion.width);
    read(motion, "height", c.motion.height);
    int pixel_threshold = c.motion.pixel_threshold;
    read(motion, "pixel_threshold", pixel_threshold);
    if(pixel_threshold < 1 || pixel_threshold > 255) throw std::invalid_argument("motion.pixel_threshold must be 1..255");
    c.motion.pixel_threshold = static_cast<uint8_t>(pixel_threshold);
    read(motion, "minimum_changed_area", c.motion.minimum_changed_area);
    read(motion, "cooldown_ms", c.motion.cooldown_ms);
    read(motion, "warmup_frames", c.motion.warmup_frames);
    read(motion, "box_margin", c.motion.box_margin);

    const json& tracker = section(root, "tracker");
    checkKeys(tracker, "tracker", {"match_iou", "match_distance", "young_hits", "still_motion_pixels", "min_hits", "confirm_score", "max_misses", "max_track_age_ms",
                                   "position_gain", "velocity_gain", "size_gain", "abrupt_velocity_change", "history_ms"});
    read(tracker, "match_iou", c.tracker.match_iou);
    read(tracker, "match_distance", c.tracker.match_distance);
    read(tracker, "young_hits", c.tracker.young_hits);
    read(tracker, "still_motion_pixels", c.tracker.still_motion_pixels);
    read(tracker, "min_hits", c.tracker.min_hits);
    read(tracker, "confirm_score", c.tracker.confirm_score);
    read(tracker, "max_misses", c.tracker.max_misses);
    read(tracker, "max_track_age_ms", c.tracker.max_track_age_ms);
    read(tracker, "position_gain", c.tracker.position_gain);
    read(tracker, "velocity_gain", c.tracker.velocity_gain);
    read(tracker, "size_gain", c.tracker.size_gain);
    read(tracker, "abrupt_velocity_change", c.tracker.abrupt_velocity_change);
    read(tracker, "history_ms", c.tracker.history_ms);

    const json& scheduler = section(root, "scheduler");
    checkKeys(scheduler, "scheduler", {"mode", "startup_frames", "fixed_interval_ms", "idle_poll_ms", "active_hold_ms",
                                       "max_semantic_age_ms", "min_semantic_interval_ms", "unexplained_motion_fraction"});
    if(scheduler.contains("mode")){
        std::string mode;
        read(scheduler, "mode", mode);
        c.scheduler.mode = schedulerModeFromString(mode);
    }
    read(scheduler, "startup_frames", c.scheduler.startup_frames);
    read(scheduler, "fixed_interval_ms", c.scheduler.fixed_interval_ms);
    read(scheduler, "idle_poll_ms", c.scheduler.idle_poll_ms);
    read(scheduler, "active_hold_ms", c.scheduler.active_hold_ms);
    read(scheduler, "max_semantic_age_ms", c.scheduler.max_semantic_age_ms);
    read(scheduler, "min_semantic_interval_ms", c.scheduler.min_semantic_interval_ms);
    read(scheduler, "unexplained_motion_fraction", c.scheduler.unexplained_motion_fraction);

    const json& pipeline = section(root, "pipeline");
    checkKeys(pipeline, "pipeline", {"source", "realtime", "async", "busy_policy", "detector", "simulated_jsonl",
                                     "simulated_delay_ms", "max_frames"});
    read(pipeline, "source", c.pipeline.source);
    read(pipeline, "realtime", c.pipeline.realtime);
    read(pipeline, "async", c.pipeline.async);
    if(pipeline.contains("busy_policy")){
        std::string policy;
        read(pipeline, "busy_policy", policy);
        c.pipeline.busy_policy = busyPolicyFromString(policy);
    }
    read(pipeline, "detector", c.pipeline.detector);
    read(pipeline, "simulated_jsonl", c.pipeline.simulated_jsonl);
    read(pipeline, "simulated_delay_ms", c.pipeline.simulated_delay_ms);
    read(pipeline, "max_frames", c.pipeline.max_frames);

    const json& camera = section(root, "camera");
    checkKeys(camera, "camera", {"width", "height", "fps", "buffers", "poll_timeout_ms"});
    read(camera, "width", c.camera.width);
    read(camera, "height", c.camera.height);
    read(camera, "fps", c.camera.fps);
    read(camera, "buffers", c.camera.buffers);
    read(camera, "poll_timeout_ms", c.camera.poll_timeout_ms);

    const json& output = section(root, "output");
    checkKeys(output, "output", {
        "directory", "save_frames_csv", "save_jsonl", "overlay", "overlay_interval_ms",
        "viewer_port", "viewer_fps"
    });
    read(output, "directory", c.output.directory);
    read(output, "save_frames_csv", c.output.save_frames_csv);
    read(output, "save_jsonl", c.output.save_jsonl);
    read(output, "overlay", c.output.overlay);
    read(output, "overlay_interval_ms", c.output.overlay_interval_ms);
    int64_t viewer_port = c.output.viewer_port;
    int64_t viewer_fps = c.output.viewer_fps;
    read(output, "viewer_port", viewer_port);
    read(output, "viewer_fps", viewer_fps);
    if(viewer_port < 0 || viewer_port > 65535) throw std::invalid_argument("output.viewer_port must be in [0, 65535]");
    if(viewer_fps < 1 || viewer_fps > 30) throw std::invalid_argument("output.viewer_fps must be in [1, 30]");
    c.output.viewer_port = static_cast<uint32_t>(viewer_port);
    c.output.viewer_fps = static_cast<uint32_t>(viewer_fps);

    validateConfig(c);
    return c;
}

RuntimeConfig loadConfig(const std::string& path){
    std::ifstream in(path);
    if(!in) throw std::invalid_argument("Could not open config file: " + path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    return parseConfig(buffer.str());
}

void validateConfig(const RuntimeConfig& c){
    auto require = [](bool ok, const char* message){
        if(!ok) throw std::invalid_argument(message);
    };

    require(c.model.confidence > 0.0f && c.model.confidence < 1.0f, "model.confidence must be in (0, 1)");
    require(c.model.iou > 0.0f && c.model.iou <= 1.0f, "model.iou must be in (0, 1]");
    require(c.model.threads >= 0, "model.threads must be >= 0");
    require(c.model.input_width > 0 && c.model.input_height > 0, "model input size must be positive");

    require(c.motion.width > 0 && c.motion.height > 0, "motion resolution must be positive");
    require(c.motion.minimum_changed_area <= c.motion.width*c.motion.height, "motion.minimum_changed_area exceeds motion resolution");
    require(c.motion.box_margin >= 0.0f, "motion.box_margin must be >= 0");

    require(c.tracker.match_iou > 0.0f && c.tracker.match_iou <= 1.0f, "tracker.match_iou must be in (0, 1]");
    require(c.tracker.min_hits >= 1, "tracker.min_hits must be >= 1");
    require(c.tracker.match_distance >= 0.0f, "tracker.match_distance must be >= 0");
    require(c.tracker.max_misses >= 1, "tracker.max_misses must be >= 1");
    require(c.tracker.position_gain > 0.0f && c.tracker.position_gain <= 1.0f, "tracker.position_gain must be in (0, 1]");
    require(c.tracker.velocity_gain >= 0.0f && c.tracker.velocity_gain <= 1.0f, "tracker.velocity_gain must be in [0, 1]");
    require(c.tracker.size_gain >= 0.0f && c.tracker.size_gain <= 1.0f, "tracker.size_gain must be in [0, 1]");
    require(c.tracker.history_ms > 0, "tracker.history_ms must be positive");

    require(c.scheduler.fixed_interval_ms > 0, "scheduler.fixed_interval_ms must be positive");
    require(c.scheduler.idle_poll_ms > 0, "scheduler.idle_poll_ms must be positive");
    require(c.scheduler.min_semantic_interval_ms <= c.scheduler.max_semantic_age_ms,
            "scheduler.min_semantic_interval_ms must not exceed max_semantic_age_ms");
    require(c.scheduler.unexplained_motion_fraction >= 0.0f && c.scheduler.unexplained_motion_fraction <= 1.0f,
            "scheduler.unexplained_motion_fraction must be in [0, 1]");

    require(c.pipeline.detector == "onnx" || c.pipeline.detector == "simulated", "pipeline.detector must be onnx or simulated");

    require(c.camera.width > 0 && c.camera.height > 0 && c.camera.width % 2 == 0, "camera size must be positive with even width");
    require(c.camera.fps > 0, "camera.fps must be positive");
    require(c.camera.buffers >= 2, "camera.buffers must be >= 2");

    require(c.output.overlay_interval_ms > 0, "output.overlay_interval_ms must be positive");
    require(c.output.viewer_port <= 65535, "output.viewer_port must be in [0, 65535]");
    require(c.output.viewer_fps >= 1 && c.output.viewer_fps <= 30, "output.viewer_fps must be in [1, 30]");
}

std::string configToJson(const RuntimeConfig& c){
    json root;
    root["model"] = {
        {"path", c.model.path}, {"confidence", c.model.confidence}, {"iou", c.model.iou},
        {"class_id", c.model.class_id}, {"threads", c.model.threads},
        {"input_width", c.model.input_width}, {"input_height", c.model.input_height}
    };
    root["motion"] = {
        {"width", c.motion.width}, {"height", c.motion.height},
        {"pixel_threshold", static_cast<int>(c.motion.pixel_threshold)},
        {"minimum_changed_area", c.motion.minimum_changed_area}, {"cooldown_ms", c.motion.cooldown_ms},
        {"warmup_frames", c.motion.warmup_frames}, {"box_margin", c.motion.box_margin}
    };
    root["tracker"] = {
        {"match_iou", c.tracker.match_iou}, {"match_distance", c.tracker.match_distance},
        {"young_hits", c.tracker.young_hits}, {"still_motion_pixels", c.tracker.still_motion_pixels},
        {"min_hits", c.tracker.min_hits}, {"confirm_score", c.tracker.confirm_score},
        {"max_misses", c.tracker.max_misses}, {"max_track_age_ms", c.tracker.max_track_age_ms},
        {"position_gain", c.tracker.position_gain}, {"velocity_gain", c.tracker.velocity_gain},
        {"size_gain", c.tracker.size_gain}, {"abrupt_velocity_change", c.tracker.abrupt_velocity_change},
        {"history_ms", c.tracker.history_ms}
    };
    root["scheduler"] = {
        {"mode", toString(c.scheduler.mode)}, {"startup_frames", c.scheduler.startup_frames},
        {"fixed_interval_ms", c.scheduler.fixed_interval_ms}, {"idle_poll_ms", c.scheduler.idle_poll_ms},
        {"active_hold_ms", c.scheduler.active_hold_ms}, {"max_semantic_age_ms", c.scheduler.max_semantic_age_ms},
        {"min_semantic_interval_ms", c.scheduler.min_semantic_interval_ms},
        {"unexplained_motion_fraction", c.scheduler.unexplained_motion_fraction}
    };
    root["pipeline"] = {
        {"source", c.pipeline.source}, {"realtime", c.pipeline.realtime}, {"async", c.pipeline.async},
        {"busy_policy", toString(c.pipeline.busy_policy)}, {"detector", c.pipeline.detector},
        {"simulated_jsonl", c.pipeline.simulated_jsonl}, {"simulated_delay_ms", c.pipeline.simulated_delay_ms},
        {"max_frames", c.pipeline.max_frames}
    };
    root["camera"] = {
        {"width", c.camera.width}, {"height", c.camera.height}, {"fps", c.camera.fps},
        {"buffers", c.camera.buffers}, {"poll_timeout_ms", c.camera.poll_timeout_ms}
    };
    root["output"] = {
        {"directory", c.output.directory}, {"save_frames_csv", c.output.save_frames_csv},
        {"save_jsonl", c.output.save_jsonl}, {"overlay", c.output.overlay},
        {"overlay_interval_ms", c.output.overlay_interval_ms},
        {"viewer_port", c.output.viewer_port}, {"viewer_fps", c.output.viewer_fps}
    };
    return root.dump(2);
}
