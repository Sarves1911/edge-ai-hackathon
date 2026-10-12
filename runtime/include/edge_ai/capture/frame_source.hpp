#pragma once

#include <edge_ai/core/frame.hpp>
#include <optional>

// Common frame-source interface for recorded replay and live cameras.
//
// nextFrame() returns std::nullopt at end of stream. The returned Frame's pixels are
// owned by the source and stay valid only until the next nextFrame() call.
class FrameSource{
    public:
        virtual std::optional<Frame> nextFrame() = 0;
        virtual ~FrameSource() = default;
};
