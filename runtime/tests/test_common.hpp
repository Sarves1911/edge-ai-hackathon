#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <optional>
#include <vector>

#include <edge_ai/capture/frame_source.hpp>
#include <edge_ai/core/frame.hpp>
#include <edge_ai/detector/detection.hpp>

#define CHECK(condition)                                      \
    do {                                                      \
        if (!(condition)) {                                   \
            std::cerr                                         \
                << "CHECK failed: "                           \
                << #condition                                 \
                << " at "                                     \
                << __FILE__                                   \
                << ":"                                        \
                << __LINE__                                   \
                << '\n';                                      \
            return 1;                                         \
        }                                                     \
    } while (false)

#define CHECK_NEAR(actual, expected, tolerance)               \
    do {                                                      \
        const double check_a_ = static_cast<double>(actual);  \
        const double check_e_ = static_cast<double>(expected);\
        if (!(std::fabs(check_a_ - check_e_) <= (tolerance))) {\
            std::cerr                                         \
                << "CHECK_NEAR failed: "                      \
                << #actual << " = " << check_a_               \
                << ", expected " << check_e_                  \
                << " +/- " << (tolerance)                     \
                << " at " << __FILE__ << ":" << __LINE__      \
                << '\n';                                      \
            return 1;                                         \
        }                                                     \
    } while (false)

constexpr uint64_t kMs = 1'000'000ULL;

// Fake in-memory source used to test FrameSource consumers.
struct FakeFrameSource : FrameSource{
    public:
        std::optional<Frame> nextFrame() override{
            if(idx >= frames.size()) return std::nullopt;
            return frames[idx++];
        }

        void addFrame(Frame frame){ frames.push_back(frame); }

    private:
        std::vector<Frame> frames;
        size_t idx = 0;
};

// Owns synthetic pixels; padding bytes are filled with 0xEE so stride bugs show up.
struct TestImage{
    std::vector<uint8_t> pixels;
    ImageView view{};
};

inline TestImage makeImage(
    size_t width,
    size_t height,
    PixelFormat format,
    size_t padding_bytes,
    const std::function<void(size_t x, size_t y, uint8_t* pixel)>& fill
){
    size_t bpp = 3;
    if(format == PixelFormat::GRAY8) bpp = 1;
    if(format == PixelFormat::YUYV) bpp = 2;

    TestImage image;
    const size_t stride = width*bpp + padding_bytes;
    image.pixels.assign(stride*height, 0xEE);

    for(size_t y=0; y<height; y++){
        for(size_t x=0; x<width; x++){
            fill(x, y, image.pixels.data() + y*stride + x*bpp);
        }
    }

    image.view = {image.pixels.data(), width, height, stride, format};
    return image;
}

inline TestImage makeSolidRGB(size_t width, size_t height, uint8_t r, uint8_t g, uint8_t b, size_t padding_bytes = 0){
    return makeImage(width, height, PixelFormat::RGB24, padding_bytes, [=](size_t, size_t, uint8_t* p){
        p[0] = r; p[1] = g; p[2] = b;
    });
}

// Gray RGB frame with an optional bright rectangle [x0, x1) x [y0, y1).
inline TestImage makeSceneRGB(size_t width, size_t height, uint8_t background, int x0 = 0, int y0 = 0, int x1 = 0, int y1 = 0, uint8_t value = 230){
    return makeImage(width, height, PixelFormat::RGB24, 0, [=](size_t x, size_t y, uint8_t* p){
        const bool inside = static_cast<int>(x) >= x0 && static_cast<int>(x) < x1 &&
                            static_cast<int>(y) >= y0 && static_cast<int>(y) < y1;
        const uint8_t v = inside ? value : background;
        p[0] = v; p[1] = v; p[2] = v;
    });
}

inline Frame makeFrame(const ImageView& view, uint64_t ts_ns, uint64_t frame_id){
    return {view, ts_ns, frame_id};
}

inline Detection makeBox(float cx, float cy, float w, float h, float score = 0.9f){
    return {cx - w/2, cy - h/2, cx + w/2, cy + h/2, score};
}

inline DetectionBatch makeBatch(uint64_t ts_ns, std::vector<Detection> detections, uint64_t frame_id = 0){
    DetectionBatch batch;
    batch.frame_ts_ns = ts_ns;
    batch.frame_id = frame_id;
    batch.detections = std::move(detections);
    return batch;
}
