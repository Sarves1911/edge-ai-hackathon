#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <edge_ai/capture/frame_source.hpp>

struct V4L2Config{
    std::string device = "/dev/video0";
    size_t width = 640;
    size_t height = 480;
    uint32_t fps = 30;
    uint32_t buffers = 4;
    int poll_timeout_ms = 1000;
    int max_poll_timeouts = 5;
};

struct V4L2Stats{
    uint64_t captured = 0;
    uint64_t dropped = 0;       // gaps in the driver sequence counter
    uint64_t poll_timeouts = 0;
    double effective_fps = 0.0;
};

// Linux USB camera source: mmap buffers, blocking poll(), DQBUF/QBUF.
//
// The returned Frame points directly into a driver buffer (YUYV). That buffer is
// requeued at the start of the next nextFrame() call, so consumers must copy what
// they need (motion downsample, tensor preprocessing) before asking for another frame.
// Inference never owns camera memory.
class V4L2Source : public FrameSource{
    public:
        explicit V4L2Source(const V4L2Config& config);
        ~V4L2Source() override;

        V4L2Source(const V4L2Source&) = delete;
        V4L2Source& operator=(const V4L2Source&) = delete;

        std::optional<Frame> nextFrame() override;

        V4L2Stats stats() const;
        size_t width() const { return width_; }
        size_t height() const { return height_; }

    private:
        struct Buffer{
            void* start = nullptr;
            size_t length = 0;
        };

        void requeuePending();
        void cleanup() noexcept;

        V4L2Config config_;
        int fd_ = -1;
        std::vector<Buffer> buffers_;
        bool streaming_ = false;

        size_t width_ = 0;
        size_t height_ = 0;
        size_t stride_ = 0;

        std::optional<uint32_t> pending_index_;
        std::optional<uint32_t> last_sequence_;

        uint64_t frame_id_ = 0;
        uint64_t first_ts_ns_ = 0;
        uint64_t last_ts_ns_ = 0;
        V4L2Stats stats_;
};
