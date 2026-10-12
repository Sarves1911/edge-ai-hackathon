#include <edge_ai/pipeline/pipeline.hpp>

#include <vector>

#include <edge_ai/core/time.hpp>

namespace {

float msBetween(uint64_t from_ns, uint64_t to_ns){
    return to_ns >= from_ns ? static_cast<float>(to_ns - from_ns) / 1e6f : 0.0f;
}

const char* submitName(SubmitResult r){
    switch(r){
        case SubmitResult::ACCEPTED: return "accepted";
        case SubmitResult::REPLACED: return "replaced";
        case SubmitResult::SKIPPED: return "skipped";
    }
    return "";
}

} // namespace

Pipeline::Pipeline(const RuntimeConfig& config, Detector& detector, Profiler& profiler)
    : config_(config),
      profiler_(profiler),
      motion_(config.motion),
      tracker_(config.tracker),
      scheduler_(config.scheduler),
      worker_(detector, config.pipeline.busy_policy, config.pipeline.async)
{
}

bool Pipeline::motionEnabled() const{
    const SchedulerMode mode = config_.scheduler.mode;
    return mode == SchedulerMode::MOTION_GATED || mode == SchedulerMode::ADAPTIVE;
}

bool Pipeline::trackingEnabled() const{
    return config_.scheduler.mode != SchedulerMode::ALWAYS;
}

void Pipeline::consumeResults(){
    while(std::optional<DetectionBatch> batch = worker_.poll()){
        const uint64_t now = now_monotonic_ns();

        DetectorRecord record;
        record.frame_id = batch->frame_id;
        record.frame_ts_ns = batch->frame_ts_ns;
        record.infer_ms = batch->infer_ms;
        record.postprocess_ms = batch->postprocess_ms;
        record.detections = batch->detections.size();

        auto it = submit_wall_ns_.find(batch->frame_id);
        if(it != submit_wall_ns_.end()) record.result_latency_ms = msBetween(it->second, now);
        // Requests up to this one are finished or were replaced.
        submit_wall_ns_.erase(submit_wall_ns_.begin(), submit_wall_ns_.upper_bound(batch->frame_id));

        profiler_.addDetectorResult(record);

        if(trackingEnabled()) tracker_.update(*batch);
        scheduler_.onDetectorResult(batch->detections.size(), batch->frame_ts_ns);

        status_.last_infer_ms = batch->infer_ms;
        last_batch_ = std::move(*batch);
        batch_arrived_this_frame_ = true;
    }
}

PipelineStatus Pipeline::step(const Frame& frame){
    const uint64_t started = now_monotonic_ns();
    const ImageView& view = frame.view;
    batch_arrived_this_frame_ = false;

    FrameRecord record;
    record.frame_id = frame.frame_id;
    record.ts_ns = frame.timestamp_ns;

    MotionResult motion;
    if(motionEnabled()){
        motion = motion_.update(view, frame.timestamp_ns);
        record.motion_ms = motion.processing_ms;
        record.motion_score = motion.score;
        record.motion_active = motion.active;
        record.motion_event = motion.event;
    }

    const uint64_t track_started = now_monotonic_ns();
    if(trackingEnabled()){
        tracker_.setFrameSize(view.width, view.height);
        if(motionEnabled()){
            const uint32_t still = config_.tracker.still_motion_pixels;
            tracker_.predict(frame.timestamp_ns, [&](const Track& t){
                return motion_.changedPixelsInside(t.box()) <= still;
            });
        }
        else{
            tracker_.predict(frame.timestamp_ns);
        }
    }
    consumeResults();
    record.track_ms = msBetween(track_started, now_monotonic_ns());

    SchedulerInputs inputs;
    inputs.ts_ns = frame.timestamp_ns;
    inputs.motion_active = motion.active;
    inputs.motion_event = motion.event;
    inputs.motion_score = motion.score;
    if(motion.active) inputs.unexplained_motion_fraction = motion_.changedFractionOutside(tracker_.boxes(false));
    inputs.confirmed_tracks = tracker_.confirmedCount();
    inputs.tentative_tracks = tracker_.tracks().size() - inputs.confirmed_tracks;
    inputs.tracker_failed = tracker_.failed();
    inputs.tracker_weak = tracker_.weak();
    inputs.abrupt_motion = tracker_.abruptMotion();
    inputs.detector_busy = worker_.async() && worker_.busy();

    const Decision decision = scheduler_.decide(inputs);

    if(decision.action == Action::RUN_DETECTOR){
        // Frame pixels are copied into a preallocated tensor here; nothing keeps them afterwards.
        submit_wall_ns_[frame.frame_id] = now_monotonic_ns();
        const SubmitResult submit = worker_.trySubmit(view, frame.timestamp_ns, frame.frame_id);
        record.submit = submitName(submit);

        if(submit == SubmitResult::SKIPPED){
            submit_wall_ns_.erase(frame.frame_id);
        }
        else{
            record.preprocess_ms = worker_.lastPreprocessMs();
            scheduler_.markDetectorRun(frame.timestamp_ns);
            tracker_.acknowledgeFailure();
        }

        // Synchronous mode produced its result inline.
        if(!worker_.async()) consumeResults();
    }

    // Output boxes: raw detector boxes without tracking (parity baseline), tracks otherwise.
    std::vector<Detection> boxes;
    std::vector<int> ids;
    if(trackingEnabled()){
        for(const Track& t : tracker_.tracks()){
            if(!t.confirmed) continue;
            boxes.push_back(t.box());
            ids.push_back(t.id);
        }
    }
    else if(last_batch_.has_value()){
        boxes = last_batch_->detections;
    }

    float result_age_ms = -1.0f;
    if(last_batch_.has_value()) result_age_ms = msBetween(last_batch_->frame_ts_ns, frame.timestamp_ns);

    const bool ran = batch_arrived_this_frame_ && last_batch_.has_value() && last_batch_->frame_id == frame.frame_id;
    profiler_.writeDetections(
        frame.frame_id, frame.timestamp_ns, view.width, view.height,
        ran, decision.reason, result_age_ms, boxes, ids
    );

    record.action = toString(decision.action);
    record.reason = decision.reason;
    record.state = toString(scheduler_.state());
    record.tracks = tracker_.tracks().size();
    record.confirmed_tracks = tracker_.confirmedCount();
    record.result_age_ms = result_age_ms;
    record.total_ms = msBetween(started, now_monotonic_ns());
    profiler_.addFrame(record);

    status_.frame_id = frame.frame_id;
    status_.ts_ns = frame.timestamp_ns;
    status_.state = scheduler_.state();
    status_.action = decision.action;
    status_.reason = decision.reason;
    status_.persons = trackingEnabled() ? tracker_.confirmedCount() : boxes.size();
    status_.detector_calls = worker_.submitted();
    status_.frames = profiler_.frames();
    status_.last_result_age_ms = result_age_ms;
    status_.motion_score = motion.score;
    return status_;
}

size_t Pipeline::run(
    FrameSource& source,
    uint64_t max_frames,
    const std::atomic<bool>* stop,
    const std::function<void(const PipelineStatus&)>& on_frame
){
    size_t processed = 0;
    while(max_frames == 0 || processed < max_frames){
        if(stop != nullptr && stop->load()) break;

        std::optional<Frame> frame = source.nextFrame();
        if(!frame.has_value()) break;

        const PipelineStatus& s = step(*frame);
        ++processed;
        if(on_frame) on_frame(s);
    }
    drain();
    return processed;
}

void Pipeline::drain(){
    worker_.waitIdle();
    consumeResults();
    profiler_.flush();
}
