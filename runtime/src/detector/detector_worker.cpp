#include <edge_ai/detector/detector_worker.hpp>

#include <stdexcept>

#include <edge_ai/core/time.hpp>

BusyPolicy busyPolicyFromString(const std::string& policy){
    if(policy == "skip") return BusyPolicy::SKIP;
    if(policy == "latest") return BusyPolicy::LATEST;
    throw std::invalid_argument("unknown busy policy: " + policy);
}

const char* toString(BusyPolicy policy){
    return policy == BusyPolicy::SKIP ? "skip" : "latest";
}

DetectorWorker::DetectorWorker(Detector& detector, BusyPolicy policy, bool async)
    : detector_(detector),
      policy_(policy),
      async_(async)
{
    // Tensor A/B: the only inference buffers for the lifetime of the worker.
    const size_t count = 3*detector_.inputWidth()*detector_.inputHeight();
    for(Slot& slot : slots_) slot.tensor.assign(count, 0.0f);

    if(async_) worker_ = std::thread(&DetectorWorker::run, this);
}

DetectorWorker::~DetectorWorker(){
    stop();
}

void DetectorWorker::stop(){
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    work_cv_.notify_all();
    idle_cv_.notify_all();
    if(worker_.joinable()) worker_.join();
}

const PreprocessMetadata& DetectorWorker::geometryFor(const ImageView& view){
    if(!have_cached_meta_ || cached_meta_.original_width != view.width || cached_meta_.original_height != view.height){
        cached_meta_ = calculateLetterboxGeometry(view.width, view.height, detector_.inputWidth(), detector_.inputHeight());
        have_cached_meta_ = true;
    }
    return cached_meta_;
}

void DetectorWorker::publish(DetectionBatch batch){
    if(result_.has_value()) ++dropped_results_;
    result_ = std::move(batch);
    ++completed_;
}

SubmitResult DetectorWorker::trySubmit(const ImageView& view, uint64_t frame_ts_ns, uint64_t frame_id){
    const PreprocessMetadata& meta = geometryFor(view);

    if(!async_){
        // Synchronous baseline: same preallocated tensor, detector inline.
        Slot& slot = slots_[0];
        const uint64_t started = now_monotonic_ns();
        preprocessInto(view, meta, slot.tensor.data());
        last_preprocess_ms_ = static_cast<float>(now_monotonic_ns() - started) / 1e6f;

        DetectionBatch batch = detector_.infer(slot.tensor.data(), meta, frame_ts_ns, frame_id);
        ++submitted_;

        std::lock_guard<std::mutex> lock(mutex_);
        publish(std::move(batch));
        return SubmitResult::ACCEPTED;
    }

    Slot* target = nullptr;
    SubmitResult outcome = SubmitResult::ACCEPTED;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if(stopping_) throw std::runtime_error("detector worker stopped");
        if(error_) std::rethrow_exception(error_);

        bool any_busy = false;
        for(Slot& slot : slots_){
            if(slot.state != SlotState::FREE) any_busy = true;
        }

        if(policy_ == BusyPolicy::SKIP && any_busy){
            ++skipped_;
            return SubmitResult::SKIPPED;
        }

        for(Slot& slot : slots_){
            if(slot.state == SlotState::FREE){ target = &slot; break; }
        }

        if(target == nullptr){
            // LATEST: overwrite the unread pending request; never queue a backlog.
            for(Slot& slot : slots_){
                if(slot.state == SlotState::PENDING){ target = &slot; break; }
            }
            if(target == nullptr){
                ++skipped_;
                return SubmitResult::SKIPPED;
            }
            ++replaced_;
            outcome = SubmitResult::REPLACED;
        }

        // WRITING keeps the worker from taking this slot while it is being filled.
        target->state = SlotState::WRITING;
    }

    const uint64_t started = now_monotonic_ns();
    try{
        preprocessInto(view, meta, target->tensor.data());
    }
    catch(...){
        std::lock_guard<std::mutex> lock(mutex_);
        target->state = SlotState::FREE;
        idle_cv_.notify_all();
        throw;
    }
    last_preprocess_ms_ = static_cast<float>(now_monotonic_ns() - started) / 1e6f;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        target->meta = meta;
        target->frame_ts_ns = frame_ts_ns;
        target->frame_id = frame_id;
        target->sequence = next_sequence_++;
        target->state = SlotState::PENDING;
    }
    ++submitted_;
    work_cv_.notify_one();
    return outcome;
}

void DetectorWorker::run(){
    while(true){
        Slot* slot = nullptr;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            work_cv_.wait(lock, [this]{
                if(stopping_) return true;
                for(const Slot& s : slots_) if(s.state == SlotState::PENDING) return true;
                return false;
            });
            if(stopping_) return;

            // Oldest pending first (there is at most one in practice).
            for(Slot& s : slots_){
                if(s.state == SlotState::PENDING && (slot == nullptr || s.sequence < slot->sequence)) slot = &s;
            }
            slot->state = SlotState::IN_FLIGHT;
        }

        try{
            DetectionBatch batch = detector_.infer(slot->tensor.data(), slot->meta, slot->frame_ts_ns, slot->frame_id);

            std::lock_guard<std::mutex> lock(mutex_);
            publish(std::move(batch));
            slot->state = SlotState::FREE;
        }
        catch(...){
            std::lock_guard<std::mutex> lock(mutex_);
            error_ = std::current_exception();
            slot->state = SlotState::FREE;
        }
        idle_cv_.notify_all();
    }
}

std::optional<DetectionBatch> DetectorWorker::poll(){
    std::lock_guard<std::mutex> lock(mutex_);
    if(error_){
        std::exception_ptr error = error_;
        error_ = nullptr;
        std::rethrow_exception(error);
    }
    std::optional<DetectionBatch> out = std::move(result_);
    result_.reset();
    return out;
}

bool DetectorWorker::busy() const{
    std::lock_guard<std::mutex> lock(mutex_);
    for(const Slot& slot : slots_){
        if(slot.state != SlotState::FREE) return true;
    }
    return false;
}

void DetectorWorker::waitIdle(){
    std::unique_lock<std::mutex> lock(mutex_);
    idle_cv_.wait(lock, [this]{
        if(stopping_ || error_) return true;
        for(const Slot& slot : slots_) if(slot.state != SlotState::FREE) return false;
        return true;
    });
}
