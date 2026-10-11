#include <cstdint>
#include <vector>

#include <edge_ai/core/frame.hpp>

#include "test_common.hpp"

int main()
{
    std::vector<uint8_t> pixels = {
        1,  2,  3,  4,  200, 200,
        5,  6,  7,  8,  200, 200,
        9, 10, 11, 12,  200, 200
    };

    ImageView view;

    view.data = pixels.data();
    view.width = 4;
    view.height = 3;
    view.stride = 6;
    view.format = PixelFormat::GRAY8;

    CHECK(view.data != nullptr);
    CHECK(view.width == 4);
    CHECK(view.height == 3);
    CHECK(view.stride == 6);
    CHECK(view.format == PixelFormat::GRAY8);

    uint8_t expected = 1;

    for (size_t y = 0; y < view.height; ++y) {

        const size_t row_start =
            y * view.stride;

        for (
            size_t x = 0;
            x < view.width;
            ++x
        ) {
            const size_t index =
                row_start + x;

            CHECK(view.data[index] == expected);

            ++expected;
        }
    }

    CHECK(expected == 13);

    Frame frame;

    frame.view = view;
    frame.frame_id = 42;
    frame.timestamp_ns = 123456789;

    CHECK(frame.frame_id == 42);
    CHECK(frame.timestamp_ns == 123456789);

    CHECK(frame.view.data == pixels.data());
    CHECK(frame.view.width == 4);
    CHECK(frame.view.height == 3);
    CHECK(frame.view.stride == 6);
    CHECK(
        frame.view.format ==
        PixelFormat::GRAY8
    );

    return 0;
}