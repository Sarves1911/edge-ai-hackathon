#pragma once

#include <cstddef>
#include <cstdint>

#include <edge_ai/detector/detection.hpp>
#include <edge_ai/preprocess/preprocess.hpp>

// Backend-neutral person detector.
//
// Input is the preprocessed 3 x inputHeight() x inputWidth() CHW float tensor plus
// the letterbox metadata needed to map boxes back to original-frame coordinates.
class Detector{
    public:
        virtual ~Detector() = default;

        virtual size_t inputWidth() const = 0;
        virtual size_t inputHeight() const = 0;

        virtual DetectionBatch infer(
            const float* chw,
            const PreprocessMetadata& meta,
            uint64_t frame_ts_ns,
            uint64_t frame_id
        ) = 0;
};
