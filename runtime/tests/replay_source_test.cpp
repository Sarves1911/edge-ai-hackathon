#include <cstdint>
#include <exception>
#include <iostream>
#include <string>

#include <edge_ai/capture/replay_source.hpp>

#include "test_common.hpp"

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr
            << "usage: replay_source_test "
            << "<test-video>\n";

        return 1;
    }

    /*
     * First verify that invalid input fails.
     */
    bool invalid_path_threw = false;

    try {
        ReplaySource invalid_source(
            "__file_that_should_not_exist__.mp4",
            ReplayMode::UNLIMITED
        );
    }
    catch (const std::exception&) {
        invalid_path_threw = true;
    }

    CHECK(invalid_path_threw);

    /*
     * Now decode a real recording.
     */
    ReplaySource source(
        argv[1],
        ReplayMode::UNLIMITED
    );

    uint64_t expected_frame_id = 0;

    bool have_previous_timestamp = false;
    uint64_t previous_timestamp = 0;

    size_t decoded_count = 0;

    while (true) {

        auto frame =
            source.nextFrame();

        if (!frame.has_value()) {
            break;
        }

        CHECK(
            frame->frame_id ==
            expected_frame_id
        );

        CHECK(frame->view.data != nullptr);

        CHECK(frame->view.width > 0);
        CHECK(frame->view.height > 0);

        CHECK(
            frame->view.format ==
            PixelFormat::RGB24
        );

        /*
         * ReplaySource currently generates tightly
         * packed RGB24 output.
         */
        CHECK(
            frame->view.stride ==
            frame->view.width * 3
        );

        if (have_previous_timestamp) {
            CHECK(
                frame->timestamp_ns >=
                previous_timestamp
            );
        }

        previous_timestamp =
            frame->timestamp_ns;

        have_previous_timestamp = true;

        ++expected_frame_id;
        ++decoded_count;
    }

    CHECK(decoded_count > 0);

    /*
     * Calling again after EOS should continue to
     * report EOS.
     */
    auto after_eof =
        source.nextFrame();

    CHECK(!after_eof.has_value());

    return 0;
}