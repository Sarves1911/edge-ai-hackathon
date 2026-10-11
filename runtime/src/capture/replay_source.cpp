#include <edge_ai/capture/replay_source.hpp>

#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>

#include <edge_ai/core/time.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/error.h>
#include <libavutil/mathematics.h>
#include <libavutil/pixfmt.h>
#include <libavutil/rational.h>
#include <libswscale/swscale.h>
}

namespace {

std::string ffmpegError(int error_code)
{
    char buffer[AV_ERROR_MAX_STRING_SIZE];

    av_strerror(
        error_code,
        buffer,
        sizeof(buffer)
    );

    return std::string(buffer);
}

} // namespace


ReplaySource::ReplaySource(
    const std::string& path,
    ReplayMode mode
)
    : mode_(mode)
{
    try {
        int ret = avformat_open_input(
            &format_ctx_,
            path.c_str(),
            nullptr,
            nullptr
        );

        if (ret < 0) {
            throw std::runtime_error(
                "Could not open video: " +
                ffmpegError(ret)
            );
        }

        ret = avformat_find_stream_info(
            format_ctx_,
            nullptr
        );

        if (ret < 0) {
            throw std::runtime_error(
                "Could not read stream information: " +
                ffmpegError(ret)
            );
        }

        // Find the first video stream.
        for (
            unsigned int i = 0;
            i < format_ctx_->nb_streams;
            ++i
        ) {
            AVStream* stream =
                format_ctx_->streams[i];

            if (
                stream->codecpar->codec_type ==
                AVMEDIA_TYPE_VIDEO
            ) {
                video_stream_index_ =
                    static_cast<int>(i);

                video_stream_ = stream;

                break;
            }
        }

        if (
            video_stream_index_ < 0 ||
            video_stream_ == nullptr
        ) {
            throw std::runtime_error(
                "No video stream found"
            );
        }

        const AVCodec* decoder =
            avcodec_find_decoder(
                video_stream_->codecpar->codec_id
            );

        if (decoder == nullptr) {
            throw std::runtime_error(
                "No decoder found for video codec"
            );
        }

        codec_ctx_ =
            avcodec_alloc_context3(decoder);

        if (codec_ctx_ == nullptr) {
            throw std::runtime_error(
                "Could not allocate decoder context"
            );
        }

        ret = avcodec_parameters_to_context(
            codec_ctx_,
            video_stream_->codecpar
        );

        if (ret < 0) {
            throw std::runtime_error(
                "Could not copy codec parameters: " +
                ffmpegError(ret)
            );
        }

        ret = avcodec_open2(
            codec_ctx_,
            decoder,
            nullptr
        );

        if (ret < 0) {
            throw std::runtime_error(
                "Could not open decoder: " +
                ffmpegError(ret)
            );
        }

        packet_ = av_packet_alloc();

        if (packet_ == nullptr) {
            throw std::runtime_error(
                "Could not allocate AVPacket"
            );
        }

        decoded_frame_ = av_frame_alloc();

        if (decoded_frame_ == nullptr) {
            throw std::runtime_error(
                "Could not allocate AVFrame"
            );
        }
    }
    catch (...) {
        cleanup();
        throw;
    }
}


ReplaySource::~ReplaySource()
{
    cleanup();
}


void ReplaySource::cleanup() noexcept
{
    if (sws_ctx_ != nullptr) {
        sws_freeContext(sws_ctx_);
        sws_ctx_ = nullptr;
    }

    if (decoded_frame_ != nullptr) {
        av_frame_free(&decoded_frame_);
    }

    if (packet_ != nullptr) {
        av_packet_free(&packet_);
    }

    if (codec_ctx_ != nullptr) {
        avcodec_free_context(&codec_ctx_);
    }

    if (format_ctx_ != nullptr) {
        avformat_close_input(&format_ctx_);
    }

    video_stream_ = nullptr;
}


void ReplaySource::pace(
    uint64_t source_timestamp_ns
)
{
    if (mode_ == ReplayMode::UNLIMITED) {
        return;
    }

    if (!machine_start_ns_.has_value()) {
        machine_start_ns_ = now_monotonic_ns();
        replay_start_ns_ = source_timestamp_ns;
        return;
    }

    if (
        source_timestamp_ns <
        *replay_start_ns_
    ) {
        throw std::runtime_error(
            "Replay timestamp moved backwards"
        );
    }

    const uint64_t actual_elapsed_ns =
        now_monotonic_ns() -
        *machine_start_ns_;

    const uint64_t source_elapsed_ns =
        source_timestamp_ns -
        *replay_start_ns_;

    if (
        actual_elapsed_ns <
        source_elapsed_ns
    ) {
        const uint64_t remaining_ns =
            source_elapsed_ns -
            actual_elapsed_ns;

        std::this_thread::sleep_for(
            std::chrono::nanoseconds(
                remaining_ns
            )
        );
    }
}


Frame ReplaySource::buildFrame()
{
    const int width =
        decoded_frame_->width;

    const int height =
        decoded_frame_->height;

    if (width <= 0 || height <= 0) {
        throw std::runtime_error(
            "Decoded frame has invalid dimensions"
        );
    }

    /*
     * Convert whatever format the decoder produced
     * (often YUV420P/NV12/etc.) into RGB24.
     */
    sws_ctx_ = sws_getCachedContext(
        sws_ctx_,

        width,
        height,
        static_cast<AVPixelFormat>(
            decoded_frame_->format
        ),

        width,
        height,
        AV_PIX_FMT_RGB24,

        SWS_BILINEAR,

        nullptr,
        nullptr,
        nullptr
    );

    if (sws_ctx_ == nullptr) {
        throw std::runtime_error(
            "Could not create FFmpeg pixel conversion context"
        );
    }

    const size_t stride =
        static_cast<size_t>(width) * 3;

    const size_t buffer_size =
        stride *
        static_cast<size_t>(height);

    rgb_buffer_.resize(buffer_size);

    uint8_t* destination_data[4] = {
        rgb_buffer_.data(),
        nullptr,
        nullptr,
        nullptr
    };

    int destination_linesize[4] = {
        static_cast<int>(stride),
        0,
        0,
        0
    };

    const int converted_rows =
        sws_scale(
            sws_ctx_,

            decoded_frame_->data,
            decoded_frame_->linesize,

            0,
            height,

            destination_data,
            destination_linesize
        );

    if (converted_rows != height) {
        throw std::runtime_error(
            "Could not convert decoded frame to RGB24"
        );
    }

    /*
     * best_effort_timestamp is usually preferable for
     * presentation-order decoded frames.
     */
    int64_t raw_timestamp =
        decoded_frame_->best_effort_timestamp;

    if (raw_timestamp == AV_NOPTS_VALUE) {
        raw_timestamp = decoded_frame_->pts;
    }

    if (raw_timestamp == AV_NOPTS_VALUE) {
        throw std::runtime_error(
            "Decoded frame has no usable timestamp"
        );
    }

    /*
     * Convert:
     *
     *     stream timestamp units
     *
     * into:
     *
     *     nanoseconds
     */
    const AVRational nanosecond_time_base = {
        1,
        1000000000
    };

    const int64_t timestamp_ns_signed =
        av_rescale_q(
            raw_timestamp,
            video_stream_->time_base,
            nanosecond_time_base
        );

    if (timestamp_ns_signed < 0) {
        throw std::runtime_error(
            "Negative video timestamp is not supported"
        );
    }

    const uint64_t timestamp_ns =
        static_cast<uint64_t>(
            timestamp_ns_signed
        );

    pace(timestamp_ns);

    ImageView view;

    view.data = rgb_buffer_.data();
    view.width =
        static_cast<size_t>(width);

    view.height =
        static_cast<size_t>(height);

    view.stride = stride;
    view.format = PixelFormat::RGB24;

    Frame frame;

    frame.view = view;
    frame.frame_id = frame_id_;
    frame.timestamp_ns = timestamp_ns;

    ++frame_id_;

    return frame;
}


std::optional<Frame>
ReplaySource::nextFrame()
{
    while (true) {

        /*
         * First ask the decoder whether it already
         * has a completed image available.
         */
        int ret = avcodec_receive_frame(
            codec_ctx_,
            decoded_frame_
        );

        if (ret == 0) {
            return buildFrame();
        }

        if (ret == AVERROR_EOF) {
            return std::nullopt;
        }

        if (ret != AVERROR(EAGAIN)) {
            throw std::runtime_error(
                "Decoder receive error: " +
                ffmpegError(ret)
            );
        }

        /*
         * The decoder needs more input.
         *
         * If the container has already reached EOF,
         * flush the decoder so it can release any
         * internally buffered frames.
         */
        if (input_eof_) {

            if (!flush_sent_) {

                ret = avcodec_send_packet(
                    codec_ctx_,
                    nullptr
                );

                if (
                    ret < 0 &&
                    ret != AVERROR_EOF
                ) {
                    throw std::runtime_error(
                        "Could not flush decoder: " +
                        ffmpegError(ret)
                    );
                }

                flush_sent_ = true;

                continue;
            }

            /*
             * Normally receive_frame() will return
             * AVERROR_EOF after the decoder is drained.
             */
            return std::nullopt;
        }

        /*
         * Read the next compressed packet from
         * the media container.
         */
        ret = av_read_frame(
            format_ctx_,
            packet_
        );

        if (ret == AVERROR_EOF) {
            input_eof_ = true;
            continue;
        }

        if (ret < 0) {
            throw std::runtime_error(
                "Could not read media packet: " +
                ffmpegError(ret)
            );
        }

        /*
         * The file may also contain audio or other
         * streams. Ignore everything except video.
         */
        if (
            packet_->stream_index !=
            video_stream_index_
        ) {
            av_packet_unref(packet_);
            continue;
        }

        ret = avcodec_send_packet(
            codec_ctx_,
            packet_
        );

        av_packet_unref(packet_);

        /*
         * Because we call receive_frame() before trying
         * to send another packet, EAGAIN should be rare,
         * but handling it is harmless.
         */
        if (ret == AVERROR(EAGAIN)) {
            continue;
        }

        if (ret < 0) {
            throw std::runtime_error(
                "Could not send packet to decoder: " +
                ffmpegError(ret)
            );
        }
    }
}