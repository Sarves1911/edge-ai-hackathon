#pragma once

#include <cstddef>
#include <cstdint>

enum class PixelFormat{
    GRAY8,
    RGB24,
    BGR24,
    YUYV
};

struct ImageView{
    const uint8_t* data;
    size_t width;
    size_t height;
    size_t stride;

    PixelFormat format;
};

struct Frame{
    ImageView view;
    uint64_t timestamp_ns;
    uint64_t frame_id;
};