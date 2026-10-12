#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <edge_ai/core/frame.hpp>
#include <edge_ai/detector/detector.hpp>
#include <edge_ai/preprocess/preprocess.hpp>

enum class BusyPolicy{
    SKIP,    // detector busy -> drop the request, keep tracking
    LATEST   // detector busy -> keep at most one pending request, newest wins
};

BusyPolicy busyPolicyFromString(const std::string& policy);
const char* toString(BusyPolicy policy);

enum class SubmitResult{
    ACCEPTED,
    REPLACED,   // overwrote an unread pending request (LATEST policy)
    SKIPPED     // detector busy (SKIP policy)
};

// Runs the detector off the capture/control thread.
//
// Two inference tensors (A/B) are allocated once at construction. trySubmit()
// preprocesses the frame directly into a free tensor, so the caller's frame memory
// (decoder or V4L2 buffer) can be released as soon as trySubmit() returns.
// There is never more than one in-flight and one pending request: no stale backlog.
//
// With async = false the detector runs inline inside trySubmit() using the same
// tensor, which gives the synchronous every-frame baseline.
class DetectorWorker{
    public:
        DetectorWorker(Detector& detector, BusyPolicy policy = BusyPolicy::LATEST, bool async = true);
        ~DetectorWorker();

        DetectorWorker(const DetectorWorker&) = delete;
        DetectorWorker& operator=(const DetectorWorker&) = delete;

        SubmitResult trySubmit(const ImageView& view, uint64_t frame_ts_ns, uint64_t frame_id);

        // Non-blocking. Returns the newest completed result, if any.
        std::optional<DetectionBatch> poll();

        bool busy() const;
        bool async() const { return async_; }

        // Blocks until no request is pending or in flight (end-of-stream drain).
        void waitIdle();
        void stop();

        float lastPreprocessMs() const { return last_preprocess_ms_.load(); }
        const float* tensorData(size_t slot) const { return slots_.at(slot).tensor.data(); }
        std::thread::id workerThreadId() const { return worker_.get_id(); }

        size_t submitted() const { return submitted_.load(); }
        size_t replaced() const { return replaced_.load(); }
        size_t skipped() const { return skipped_.load(); }
        size_t completed() const { return completed_.load(); }
        size_t droppedResults() const { return dropped_results_.load(); }

    private:
        enum class SlotState{ FREE, WRITING, PENDING, IN_FLIGHT };

        struct Slot{
            std::vector<float> tensor;
            PreprocessMetadata meta{};
            uint64_t frame_ts_ns = 0;
            uint64_t frame_id = 0;
            uint64_t sequence = 0;
            SlotState state = SlotState::FREE;
        };

        void run();
        const PreprocessMetadata& geometryFor(const ImageView& view);
        void publish(DetectionBatch batch);

        Detector& detector_;
        BusyPolicy policy_;
        bool async_;

        std::array<Slot, 2> slots_;

        mutable std::mutex mutex_;
        std::condition_variable work_cv_;
        std::condition_variable idle_cv_;
        bool stopping_ = false;
        uint64_t next_sequence_ = 0;

        std::optional<DetectionBatch> result_;
        std::exception_ptr error_;

        PreprocessMetadata cached_meta_{};
        bool have_cached_meta_ = false;

        std::atomic<float> last_preprocess_ms_{0.0f};
        std::atomic<size_t> submitted_{0};
        std::atomic<size_t> replaced_{0};
        std::atomic<size_t> skipped_{0};
        std::atomic<size_t> completed_{0};
        std::atomic<size_t> dropped_results_{0};

        std::thread worker_;
};
