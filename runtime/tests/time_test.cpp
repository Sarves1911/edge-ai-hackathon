#include <chrono>
#include <cstdint>
#include <thread>

#include <edge_ai/core/time.hpp>

#include "test_common.hpp"

int main()
{
    const uint64_t t1 =
        now_monotonic_ns();

    std::this_thread::sleep_for(
        std::chrono::milliseconds(10)
    );

    const uint64_t t2 =
        now_monotonic_ns();

    CHECK(t2 > t1);

    const uint64_t elapsed_ns =
        t2 - t1;

    /*
     * We asked for 10 ms.
     *
     * Do not require exactly 10 ms because OS
     * scheduling is not exact.
     */
    CHECK(
        elapsed_ns >= 5'000'000
    );

    return 0;
}