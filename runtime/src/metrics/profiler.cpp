#include <edge_ai/metrics/profiler.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <stdexcept>

#include <sys/resource.h>

#include <nlohmann/json.hpp>

#include <edge_ai/core/time.hpp>

using nlohmann::json;

float percentile(std::vector<float> values, float percent){
    if(values.empty()) return 0.0f;
    if(percent < 0.0f || percent > 100.0f) throw std::invalid_argument("percent must be between 0 and 100");

    std::sort(values.begin(), values.end());
    if(values.size() == 1) return values[0];

    const double position = (values.size() - 1)*static_cast<double>(percent)/100.0;
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, values.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return static_cast<float>(values[lower]*(1.0 - fraction) + values[upper]*fraction);
}

ResourceUsage sampleResourceUsage(){
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);

    ResourceUsage out;
    out.cpu_user_s = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec/1e6;
    out.cpu_system_s = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec/1e6;
#if defined(__APPLE__)
    out.peak_rss_mb = static_cast<double>(usage.ru_maxrss) / (1024.0*1024.0);   // bytes
#else
    out.peak_rss_mb = static_cast<double>(usage.ru_maxrss) / 1024.0;            // kilobytes
#endif
    return out;
}

namespace {

double mean(const std::vector<float>& values){
    if(values.empty()) return 0.0;
    double sum = 0.0;
    for(float v : values) sum += v;
    return sum / static_cast<double>(values.size());
}

json stats(const std::vector<float>& values){
    return {
        {"count", values.size()},
        {"mean", mean(values)},
        {"p50", percentile(values, 50.0f)},
        {"p95", percentile(values, 95.0f)},
        {"max", values.empty() ? 0.0f : *std::max_element(values.begin(), values.end())}
    };
}

} // namespace

void Profiler::openOutputs(const std::string& directory, bool frames_csv, bool detections_jsonl){
    if(frames_csv){
        frames_csv_.open(directory + "/frames.csv");
        if(!frames_csv_) throw std::runtime_error("Could not open frames.csv in " + directory);
        frames_csv_
            << "frame_id,timestamp_ms,motion_ms,motion_score,motion_active,motion_event,"
            << "preprocess_ms,track_ms,total_ms,action,reason,state,submit,tracks,confirmed_tracks,result_age_ms\n";
    }
    if(detections_jsonl){
        detections_jsonl_.open(directory + "/detections.jsonl");
        if(!detections_jsonl_) throw std::runtime_error("Could not open detections.jsonl in " + directory);
    }
}

void Profiler::start(){
    started_ns_ = now_monotonic_ns();
}

void Profiler::addFrame(const FrameRecord& r){
    if(started_ns_ == 0) start();
    if(frames_ == 0) first_ts_ns_ = r.ts_ns;
    last_ts_ns_ = r.ts_ns;
    ++frames_;

    frame_ms_.push_back(r.total_ms);
    motion_ms_.push_back(r.motion_ms);
    if(r.preprocess_ms > 0.0f) preprocess_ms_.push_back(r.preprocess_ms);
    if(r.result_age_ms >= 0.0f) result_age_ms_.push_back(r.result_age_ms);
    if(r.motion_active) ++motion_active_frames_;
    if(r.motion_event) ++motion_events_;

    ++actions_[r.action];
    ++states_[r.state];
    if(r.action == "RUN_DETECTOR") ++reasons_[r.reason];
    if(!r.submit.empty()) ++submits_[r.submit];

    if(frames_csv_.is_open()){
        frames_csv_ << r.frame_id << ',' << std::fixed << std::setprecision(3)
            << r.ts_ns/1e6 << ','
            << r.motion_ms << ',' << std::setprecision(5) << r.motion_score << ','
            << (r.motion_active ? 1 : 0) << ',' << (r.motion_event ? 1 : 0) << ','
            << std::setprecision(3) << r.preprocess_ms << ',' << r.track_ms << ',' << r.total_ms << ','
            << r.action << ',' << r.reason << ',' << r.state << ',' << r.submit << ','
            << r.tracks << ',' << r.confirmed_tracks << ',' << r.result_age_ms << '\n';
    }
}

void Profiler::addDetectorResult(const DetectorRecord& r){
    infer_ms_.push_back(r.infer_ms);
    postprocess_ms_.push_back(r.postprocess_ms);
    result_latency_ms_.push_back(r.result_latency_ms);
    total_detections_ += r.detections;
}

void Profiler::writeDetections(
    uint64_t frame_id,
    uint64_t ts_ns,
    size_t width,
    size_t height,
    bool ran,
    const std::string& reason,
    float result_age_ms,
    const std::vector<Detection>& detections,
    const std::vector<int>& track_ids
){
    if(!detections_jsonl_.is_open()) return;

    json dets = json::array();
    for(size_t i=0; i<detections.size(); i++){
        const Detection& d = detections[i];
        json record = {
            {"class_id", 0},
            {"class_name", "person"},
            {"confidence", d.score},
            {"bbox_xyxy", {d.x1, d.y1, d.x2, d.y2}}
        };
        if(i < track_ids.size()) record["track_id"] = track_ids[i];
        dets.push_back(record);
    }

    json line = {
        {"frame_index", frame_id},
        {"timestamp_ms", ts_ns/1e6},
        {"frame", {{"width", width}, {"height", height}}},
        {"inference", {
            {"ran", ran},
            {"reason", reason},
            {"result_age_ms", result_age_ms >= 0.0f ? json(result_age_ms) : json(nullptr)}
        }},
        {"detections", dets}
    };
    detections_jsonl_ << line.dump() << '\n';
}

void Profiler::setModelInfo(const std::string& path, double size_mb, float load_ms){
    model_path_ = path;
    model_size_mb_ = size_mb;
    model_load_ms_ = load_ms;
}

void Profiler::setExtra(const std::string& key, double value){
    extra_[key] = value;
}

std::string Profiler::summaryJson() const{
    const double wall_s = started_ns_ == 0 ? 0.0 : (now_monotonic_ns() - started_ns_)/1e9;
    const double source_s = frames_ > 1 ? (last_ts_ns_ - first_ts_ns_)/1e9 : 0.0;
    const ResourceUsage usage = sampleResourceUsage();

    const size_t calls = infer_ms_.size();
    const size_t avoided = frames_ > calls ? frames_ - calls : 0;

    json summary = {
        {"total_frames", frames_},
        {"total_detector_calls", calls},
        {"detector_calls_avoided", avoided},
        {"detector_calls_avoided_pct", frames_ > 0 ? 100.0*avoided/frames_ : 0.0},
        {"detector_calls_per_minute", source_s > 0.0 ? calls/(source_s/60.0) : 0.0},
        {"inference_ms", stats(infer_ms_)},
        {"postprocess_ms", stats(postprocess_ms_)},
        {"preprocess_ms", stats(preprocess_ms_)},
        {"result_latency_ms", stats(result_latency_ms_)},
        {"frame_ms", stats(frame_ms_)},
        {"motion_ms", stats(motion_ms_)},
        {"result_age_ms", stats(result_age_ms_)},
        {"average_inference_ms", mean(infer_ms_)},
        {"p50_inference_ms", percentile(infer_ms_, 50.0f)},
        {"p95_inference_ms", percentile(infer_ms_, 95.0f)},
        {"wall_s", wall_s},
        {"source_duration_s", source_s},
        {"throughput_fps", wall_s > 0.0 ? frames_/wall_s : 0.0},
        {"peak_ram_mb", usage.peak_rss_mb},
        {"cpu_user_s", usage.cpu_user_s},
        {"cpu_system_s", usage.cpu_system_s},
        {"cpu_utilization_pct", wall_s > 0.0 ? 100.0*(usage.cpu_user_s + usage.cpu_system_s)/wall_s : 0.0},
        {"motion_active_frames", motion_active_frames_},
        {"motion_events", motion_events_},
        {"total_detections", total_detections_},
        {"inference_reasons", reasons_},
        {"actions", actions_},
        {"states", states_},
        {"submits", submits_},
        {"model", {{"path", model_path_}, {"size_mb", model_size_mb_}, {"load_ms", model_load_ms_}}}
    };
    for(const auto& [key, value] : extra_) summary[key] = value;
    return summary.dump(2);
}

void Profiler::writeSummary(const std::string& path) const{
    std::ofstream out(path);
    if(!out) throw std::runtime_error("Could not write summary: " + path);
    out << summaryJson() << '\n';
}

void Profiler::flush(){
    if(frames_csv_.is_open()) frames_csv_.flush();
    if(detections_jsonl_.is_open()) detections_jsonl_.flush();
}
