#include <cstdlib>
#include <iostream>

#include <edge_ai/capture/v4l2_source.hpp>

#include "test_common.hpp"

// Requires a camera or v4l2loopback device: EDGE_AI_V4L2_DEVICE=/dev/videoN.
// Skipped (CTest return code 77) when no device is configured.
int main(){
    const char* device = std::getenv("EDGE_AI_V4L2_DEVICE");
    if(device == nullptr){
        std::cout << "EDGE_AI_V4L2_DEVICE not set; skipping\n";
        return 77;
    }

    V4L2Config config;
    config.device = device;
    V4L2Source source(config);

    uint64_t previous_ts = 0;
    for(uint64_t i=0; i<60; i++){
        auto frame = source.nextFrame();
        CHECK(frame.has_value());
        CHECK(frame->frame_id == i);
        CHECK(frame->view.format == PixelFormat::YUYV);
        CHECK(frame->view.width == source.width());
        CHECK(frame->view.stride >= frame->view.width*2);
        if(i > 0) CHECK(frame->timestamp_ns > previous_ts);
        previous_ts = frame->timestamp_ns;
    }

    const V4L2Stats stats = source.stats();
    std::cout << "captured " << stats.captured << ", dropped " << stats.dropped
              << ", effective fps " << stats.effective_fps << '\n';
    CHECK(stats.captured == 60);
    CHECK(stats.effective_fps > 0.0);
    return 0;
}
