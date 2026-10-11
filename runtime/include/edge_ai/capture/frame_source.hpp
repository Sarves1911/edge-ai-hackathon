#pragma once

#include <edge_ai/core/frame.hpp>
#include <edge_ai/core/time.hpp>
#include <optional>
#include <vector>

enum class ReplayMode{
    REALTIME,
    UNLIMITED
};

class FrameSource{
    public:
        virtual std::optional<Frame> nextFrame() = 0;
        virtual ~FrameSource() = default;
};

struct FakeFrameSource : FrameSource{
    public:
        std::optional<Frame> nextFrame() override{
            if(idx>=frames.size()) return std::nullopt;

            auto curr = frames[idx];
            idx++;
            return curr;
        };

        void addFrame(Frame frame){ frames.push_back(frame);}
    private:
        std::vector<Frame> frames;
        size_t idx = 0;

};