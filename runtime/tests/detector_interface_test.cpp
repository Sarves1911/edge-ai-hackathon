#include <iostream>
#include <vector>

#include <edge_ai/detector/detector.hpp>
#include <edge_ai/detector/simulated_detector.hpp>

#include "test_common.hpp"

namespace {

// A second, independent implementation: the consumer must not care which one it gets.
class ConstantDetector : public Detector{
    public:
        size_t inputWidth() const override { return 8; }
        size_t inputHeight() const override { return 8; }

        DetectionBatch infer(const float*, const PreprocessMetadata&, uint64_t ts, uint64_t id) override{
            return makeBatch(ts, {{1, 2, 3, 4, 0.5f}}, id);
        }
};

// Consumer that depends only on the Detector interface.
DetectionBatch runOnce(Detector& detector, const ImageView& view, uint64_t ts, uint64_t id){
    const PreprocessResult pre = preprocess(view, detector.inputWidth(), detector.inputHeight());
    return detector.infer(pre.tensor.data(), pre.meta, ts, id);
}

} // namespace

int main(){
    TestImage image = makeSolidRGB(16, 12, 50, 60, 70);

    SimulatedDetector simulated(8, 8, [](uint64_t ts){
        return std::vector<Detection>{makeBox(5, 5, 4, 6, ts > 100*kMs ? 0.9f : 0.4f)};
    });
    ConstantDetector constant;

    const DetectionBatch a = runOnce(simulated, image.view, 250*kMs, 7);
    CHECK(a.frame_ts_ns == 250*kMs);
    CHECK(a.frame_id == 7);
    CHECK(a.detections.size() == 1);
    CHECK_NEAR(a.detections[0].score, 0.9f, 1e-6);
    CHECK(simulated.calls() == 1);

    const DetectionBatch b = runOnce(constant, image.view, 42*kMs, 3);
    CHECK(b.frame_ts_ns == 42*kMs);
    CHECK(b.frame_id == 3);
    CHECK(b.detections.size() == 1);

    // Timeline lookup returns the nearest cached entry for each source timestamp.
    SimulatedDetector cached(8, 8, SimulatedDetector::timelineScript({
        {0, {}},
        {100*kMs, {makeBox(10, 10, 4, 4)}},
        {200*kMs, {makeBox(20, 20, 4, 4), makeBox(40, 40, 4, 4)}},
    }));
    CHECK(runOnce(cached, image.view, 10*kMs, 0).detections.empty());
    CHECK(runOnce(cached, image.view, 120*kMs, 1).detections.size() == 1);
    CHECK(runOnce(cached, image.view, 190*kMs, 2).detections.size() == 2);
    CHECK(runOnce(cached, image.view, 900*kMs, 3).detections.size() == 2);

    // Injected latency is reflected in infer_ms, and the batch still carries the source ts.
    SimulatedDetector slow(8, 8, [](uint64_t){ return std::vector<Detection>{}; }, 30);
    const DetectionBatch s = runOnce(slow, image.view, 5*kMs, 9);
    CHECK(s.infer_ms >= 29.0f);
    CHECK(s.frame_ts_ns == 5*kMs);

    std::cout << "All detector interface tests passed.\n";
    return 0;
}
