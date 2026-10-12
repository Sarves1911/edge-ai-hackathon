#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <edge_ai/detector/detector.hpp>
#include <edge_ai/detector/postprocess.hpp>

// ONNX Runtime (CPUExecutionProvider) YOLOv8 person detector.
// ONNX Runtime types stay inside the .cpp so they never leak through this header.
class OnnxDetector : public Detector{
    public:
        OnnxDetector(const std::string& model_path, const PostprocessConfig& postprocess, int intra_op_threads = 0);
        ~OnnxDetector() override;

        OnnxDetector(const OnnxDetector&) = delete;
        OnnxDetector& operator=(const OnnxDetector&) = delete;

        size_t inputWidth() const override;
        size_t inputHeight() const override;

        DetectionBatch infer(const float* chw, const PreprocessMetadata& meta, uint64_t frame_ts_ns, uint64_t frame_id) override;

        float modelLoadMs() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
};
