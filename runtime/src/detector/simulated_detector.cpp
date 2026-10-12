#include <edge_ai/detector/simulated_detector.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <thread>

#include <nlohmann/json.hpp>

#include <edge_ai/core/time.hpp>

SimulatedDetector::SimulatedDetector(size_t input_width, size_t input_height, Script script, uint32_t delay_ms)
    : input_width_(input_width),
      input_height_(input_height),
      script_(std::move(script)),
      delay_ms_(delay_ms)
{
    if(input_width_ == 0 || input_height_ == 0) throw std::invalid_argument("zero detector input size");
    if(!script_) throw std::invalid_argument("empty detector script");
}

SimulatedDetector::Script SimulatedDetector::timelineScript(Timeline timeline){
    std::sort(timeline.begin(), timeline.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
    auto shared = std::make_shared<Timeline>(std::move(timeline));

    return [shared](uint64_t ts) -> std::vector<Detection>{
        const Timeline& tl = *shared;
        if(tl.empty()) return {};

        auto it = std::lower_bound(tl.begin(), tl.end(), ts, [](const auto& entry, uint64_t value){
            return entry.first < value;
        });
        if(it == tl.end()) return tl.back().second;
        if(it == tl.begin()) return it->second;

        auto prev = std::prev(it);
        return (ts - prev->first) <= (it->first - ts) ? prev->second : it->second;
    };
}

SimulatedDetector::Timeline SimulatedDetector::loadJsonl(const std::string& path){
    std::ifstream in(path);
    if(!in) throw std::runtime_error("Could not open detections JSONL: " + path);

    Timeline timeline;
    std::string line;
    while(std::getline(in, line)){
        if(line.empty()) continue;
        const nlohmann::json record = nlohmann::json::parse(line);

        const double ts_ms = record.at("timestamp_ms").get<double>();
        std::vector<Detection> detections;
        for(const auto& d : record.value("detections", nlohmann::json::array())){
            if(d.value("class_id", 0) != 0) continue;
            const auto& box = d.at("bbox_xyxy");
            detections.push_back({
                box.at(0).get<float>(), box.at(1).get<float>(),
                box.at(2).get<float>(), box.at(3).get<float>(),
                d.at("confidence").get<float>()
            });
        }
        timeline.emplace_back(static_cast<uint64_t>(std::llround(ts_ms*1e6)), std::move(detections));
    }

    return timeline;
}

DetectionBatch SimulatedDetector::infer(const float* /*chw*/, const PreprocessMetadata& /*meta*/, uint64_t frame_ts_ns, uint64_t frame_id){
    const uint64_t started = now_monotonic_ns();

    const uint32_t delay = delay_ms_.load();
    if(delay > 0) std::this_thread::sleep_for(std::chrono::milliseconds(delay));

    DetectionBatch batch;
    batch.frame_ts_ns = frame_ts_ns;
    batch.frame_id = frame_id;
    batch.detections = script_(frame_ts_ns);
    batch.infer_ms = static_cast<float>(now_monotonic_ns() - started) / 1e6f;
    batch.postprocess_ms = 0.0f;

    ++calls_;
    return batch;
}
