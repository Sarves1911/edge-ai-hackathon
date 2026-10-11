#include <chrono>
#include <cstdint>
#include <iostream>

#include <edge_ai/capture/replay_source.hpp>
#include <edge_ai/core/time.hpp>

#include "test_common.hpp"

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr
            << "usage: replay_realtime_test "
            << "<test-video>\n";

        return 1;
    }

    ReplaySource source(
        argv[1],
        ReplayMode::REALTIME
    );

    auto first =
        source.nextFrame();

    CHECK(first.has_value());

    const uint64_t source_start =
        first->timestamp_ns;

    const uint64_t machine_start =
        now_monotonic_ns();

    std::optional<Frame> current =
        first;

    /*
     * Read a few frames rather than the entire video.
     */
    for (int i = 0; i < 5; ++i) {

        current =
            source.nextFrame();

        if (!current.has_value()) {
            break;
        }
    }

    CHECK(current.has_value());

    const uint64_t machine_end =
        now_monotonic_ns();

    const uint64_t source_elapsed =
        current->timestamp_ns -
        source_start;

    const uint64_t machine_elapsed =
        machine_end -
        machine_start;

    /*
     * Real-time replay should not return the frames
     * substantially earlier than their source timeline.
     *
     * 5 ms tolerance avoids tiny scheduler/timer
     * differences.
     */
    constexpr uint64_t tolerance_ns =
        5'000'000;

    CHECK(
        machine_elapsed + tolerance_ns >=
        source_elapsed
    );

    return 0;
}