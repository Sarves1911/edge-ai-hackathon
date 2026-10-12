#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <edge_ai/capture/frame_source.hpp>

// FFmpeg forward declarations.
// Keeps FFmpeg headers out of this public header.
struct AVFormatContext;
struct AVCodecContext;
struct AVPacket;
struct AVFrame;
struct AVStream;
struct SwsContext;

enum class ReplayMode{
    REALTIME,
    UNLIMITED
};

class ReplaySource : public FrameSource {
public:
    explicit ReplaySource(
        const std::string& path,
        ReplayMode mode = ReplayMode::UNLIMITED
    );

    ~ReplaySource() override;

    ReplaySource(const ReplaySource&) = delete;
    ReplaySource& operator=(const ReplaySource&) = delete;

    std::optional<Frame> nextFrame() override;

private:
    Frame buildFrame();
    void pace(uint64_t source_timestamp_ns);

    void cleanup() noexcept;

    AVFormatContext* format_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    AVStream* video_stream_ = nullptr;

    AVPacket* packet_ = nullptr;
    AVFrame* decoded_frame_ = nullptr;

    SwsContext* sws_ctx_ = nullptr;

    int video_stream_index_ = -1;

    // Owns the RGB pixels returned through ImageView.
    std::vector<uint8_t> rgb_buffer_;

    ReplayMode mode_;

    uint64_t frame_id_ = 0;

    // Used only for REALTIME replay.
    std::optional<uint64_t> machine_start_ns_;
    std::optional<uint64_t> replay_start_ns_;

    bool input_eof_ = false;
    bool flush_sent_ = false;
};