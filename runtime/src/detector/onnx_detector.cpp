#include <edge_ai/detector/onnx_detector.hpp>

#include <array>
#include <stdexcept>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include <edge_ai/core/time.hpp>

struct OnnxDetector::Impl{
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "edge_ai"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::string input_name;
    std::string output_name;

    std::array<int64_t, 4> input_shape{};
    size_t num_attrs = 0;
    size_t num_candidates = 0;

    PostprocessConfig postprocess;
    float load_ms = 0.0f;
};

OnnxDetector::OnnxDetector(const std::string& model_path, const PostprocessConfig& postprocess, int intra_op_threads)
    : impl_(std::make_unique<Impl>())
{
    const uint64_t started = now_monotonic_ns();

    impl_->postprocess = postprocess;
    if(intra_op_threads > 0) impl_->options.SetIntraOpNumThreads(intra_op_threads);
    impl_->options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path.c_str(), impl_->options);
    Ort::Session& session = *impl_->session;

    if(session.GetInputCount() != 1 || session.GetOutputCount() < 1){
        throw std::runtime_error("Expected a single-input YOLO model");
    }

    Ort::AllocatorWithDefaultOptions allocator;
    impl_->input_name = session.GetInputNameAllocated(0, allocator).get();
    impl_->output_name = session.GetOutputNameAllocated(0, allocator).get();

    // Verify static [1, 3, H, W] float input.
    // TypeInfo must outlive the shape-info view taken from it.
    const Ort::TypeInfo input_type = session.GetInputTypeInfo(0);
    const auto input_info = input_type.GetTensorTypeAndShapeInfo();
    if(input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT){
        throw std::runtime_error("Model input must be float32");
    }
    const std::vector<int64_t> in_shape = input_info.GetShape();
    if(in_shape.size() != 4 || in_shape[1] != 3 || in_shape[2] <= 0 || in_shape[3] <= 0){
        throw std::runtime_error("Model input must be static [1, 3, H, W]");
    }
    impl_->input_shape = {1, 3, in_shape[2], in_shape[3]};

    // Verify [1, 4 + classes, N] output.
    const Ort::TypeInfo output_type = session.GetOutputTypeInfo(0);
    const auto output_info = output_type.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> out_shape = output_info.GetShape();
    if(out_shape.size() != 3 || out_shape[1] < 5 || out_shape[2] <= 0){
        throw std::runtime_error("Model output must be static [1, 4 + classes, N]");
    }
    impl_->num_attrs = static_cast<size_t>(out_shape[1]);
    impl_->num_candidates = static_cast<size_t>(out_shape[2]);

    if(postprocess.class_id >= impl_->num_attrs - 4){
        throw std::runtime_error("Configured class id is not produced by the model");
    }

    impl_->load_ms = static_cast<float>(now_monotonic_ns() - started) / 1e6f;
}

OnnxDetector::~OnnxDetector() = default;

size_t OnnxDetector::inputWidth() const { return static_cast<size_t>(impl_->input_shape[3]); }
size_t OnnxDetector::inputHeight() const { return static_cast<size_t>(impl_->input_shape[2]); }
float OnnxDetector::modelLoadMs() const { return impl_->load_ms; }

DetectionBatch OnnxDetector::infer(const float* chw, const PreprocessMetadata& meta, uint64_t frame_ts_ns, uint64_t frame_id){
    if(meta.target_width != inputWidth() || meta.target_height != inputHeight()){
        throw std::invalid_argument("Preprocessed tensor size does not match model input");
    }

    const size_t count = 3*inputWidth()*inputHeight();

    // Wraps the caller's preallocated tensor; ONNX Runtime does not copy it.
    Ort::Value input = Ort::Value::CreateTensor<float>(
        impl_->memory_info,
        const_cast<float*>(chw),
        count,
        impl_->input_shape.data(),
        impl_->input_shape.size()
    );

    const char* input_names[] = {impl_->input_name.c_str()};
    const char* output_names[] = {impl_->output_name.c_str()};

    const uint64_t infer_started = now_monotonic_ns();
    std::vector<Ort::Value> outputs = impl_->session->Run(
        Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1
    );
    const uint64_t infer_done = now_monotonic_ns();

    DetectionBatch batch;
    batch.frame_ts_ns = frame_ts_ns;
    batch.frame_id = frame_id;
    batch.detections = postprocessYolov8(
        outputs[0].GetTensorData<float>(),
        impl_->num_attrs,
        impl_->num_candidates,
        impl_->postprocess,
        meta
    );
    batch.infer_ms = static_cast<float>(infer_done - infer_started) / 1e6f;
    batch.postprocess_ms = static_cast<float>(now_monotonic_ns() - infer_done) / 1e6f;
    return batch;
}
