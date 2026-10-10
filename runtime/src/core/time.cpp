#include <edge_ai/core/time.hpp>
#include <chrono>

uint64_t now_monotonic_ns(){
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}