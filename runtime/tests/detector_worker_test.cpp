#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

#include <edge_ai/core/time.hpp>
#include <edge_ai/detector/detector_worker.hpp>
#include <edge_ai/detector/simulated_detector.hpp>

#include "test_common.hpp"

namespace {

// Records which thread ran inference.
class ThreadRecordingDetector : public Detector{
    public:
        explicit ThreadRecordingDetector(uint32_t delay_ms) : delay_ms_(delay_ms) {}

        size_t inputWidth() const override { return 32; }
        size_t inputHeight() const override { return 32; }

        DetectionBatch infer(const float* chw, const PreprocessMetadata&, uint64_t ts, uint64_t id) override{
            thread_id = std::this_thread::get_id();
            last_tensor = chw;
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms_));
            ++calls;
            return makeBatch(ts, {makeBox(10, 10, 4, 4)}, id);
        }

        std::atomic<std::thread::id> thread_id{};
        std::atomic<const float*> last_tensor{nullptr};
        std::atomic<int> calls{0};

    private:
        uint32_t delay_ms_;
};

} // namespace

int main(){
    TestImage image = makeSolidRGB(64, 48, 10, 20, 30);

    // Async: submit returns quickly while inference runs on another thread.
    {
        ThreadRecordingDetector detector(100);
        DetectorWorker worker(detector, BusyPolicy::LATEST, true);

        const float* tensor_a = worker.tensorData(0);
        const float* tensor_b = worker.tensorData(1);

        const uint64_t started = now_monotonic_ns();
        CHECK(worker.trySubmit(image.view, 1*kMs, 1) == SubmitResult::ACCEPTED);
        const float submit_ms = (now_monotonic_ns() - started)/1e6f;
        CHECK(submit_ms < 20.0f);
        CHECK(worker.busy());
        CHECK(!worker.poll().has_value());

        worker.waitIdle();
        CHECK(!worker.busy());
        const auto result = worker.poll();
        CHECK(result.has_value());
        CHECK(result->frame_ts_ns == 1*kMs);
        CHECK(result->frame_id == 1);
        CHECK(detector.thread_id.load() != std::this_thread::get_id());
        CHECK(detector.thread_id.load() == worker.workerThreadId());

        // Inference read one of the two preallocated tensors, which never moved.
        CHECK(detector.last_tensor.load() == tensor_a || detector.last_tensor.load() == tensor_b);
        CHECK(worker.tensorData(0) == tensor_a && worker.tensorData(1) == tensor_b);
    }

    // LATEST: a flood of submits while busy never forms a queue; the newest pending frame wins.
    {
        ThreadRecordingDetector detector(50);
        DetectorWorker worker(detector, BusyPolicy::LATEST, true);

        const uint64_t started = now_monotonic_ns();
        for(uint64_t i=0; i<1000; i++) worker.trySubmit(image.view, i*kMs, i);
        const float total_ms = (now_monotonic_ns() - started)/1e6f;

        // 1000 submits must not wait for 1000 inferences.
        CHECK(total_ms < 1000.0f);
        CHECK(worker.replaced() > 900);

        worker.waitIdle();
        CHECK(detector.calls.load() <= 3);

        // Results report the newest frame that actually ran, never a stale backlog.
        const auto last = worker.poll();
        CHECK(last.has_value());
        CHECK(last->frame_id == 999);
        CHECK(worker.submitted() == 1000);
    }

    // SKIP: requests while busy are refused.
    {
        ThreadRecordingDetector detector(60);
        DetectorWorker worker(detector, BusyPolicy::SKIP, true);

        CHECK(worker.trySubmit(image.view, 0, 0) == SubmitResult::ACCEPTED);
        CHECK(worker.trySubmit(image.view, 1*kMs, 1) == SubmitResult::SKIPPED);
        CHECK(worker.trySubmit(image.view, 2*kMs, 2) == SubmitResult::SKIPPED);
        CHECK(worker.skipped() == 2);

        worker.waitIdle();
        const auto result = worker.poll();
        CHECK(result.has_value() && result->frame_id == 0);
        CHECK(detector.calls.load() == 1);
    }

    // Sync mode: detector runs inline on the caller's thread (every-frame baseline).
    {
        ThreadRecordingDetector detector(0);
        DetectorWorker worker(detector, BusyPolicy::LATEST, false);
        CHECK(worker.trySubmit(image.view, 7*kMs, 7) == SubmitResult::ACCEPTED);
        CHECK(detector.thread_id.load() == std::this_thread::get_id());
        const auto result = worker.poll();
        CHECK(result.has_value() && result->frame_ts_ns == 7*kMs);
        CHECK(!worker.busy());
    }

    // Detector exceptions surface on the control thread instead of killing the worker silently.
    {
        SimulatedDetector failing(16, 16, [](uint64_t) -> std::vector<Detection>{
            throw std::runtime_error("backend failure");
        });
        DetectorWorker worker(failing, BusyPolicy::LATEST, true);
        worker.trySubmit(image.view, 0, 0);
        worker.waitIdle();
        bool threw = false;
        try{ worker.poll(); } catch(const std::runtime_error&){ threw = true; }
        CHECK(threw);
    }

    // Stop/destroy while inference is in flight joins cleanly.
    {
        ThreadRecordingDetector detector(80);
        DetectorWorker worker(detector, BusyPolicy::LATEST, true);
        worker.trySubmit(image.view, 0, 0);
        worker.stop();
        CHECK(worker.workerThreadId() == std::thread::id());
    }

    std::cout << "All detector worker tests passed.\n";
    return 0;
}
