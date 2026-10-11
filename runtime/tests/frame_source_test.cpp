#include <cstdint>
#include <optional>
#include <vector>

#include <edge_ai/capture/frame_source.hpp>
#include <edge_ai/core/frame.hpp>

#include "test_common.hpp"

int main()
{
    /*
     * These vectors own the memory.
     *
     * The Frames below only view it.
     */
    std::vector<uint8_t> pixels0 = {
        10, 11,
        12, 13
    };

    std::vector<uint8_t> pixels1 = {
        20, 21,
        22, 23
    };

    std::vector<uint8_t> pixels2 = {
        30, 31,
        32, 33
    };

    ImageView view0;
    view0.data = pixels0.data();
    view0.width = 2;
    view0.height = 2;
    view0.stride = 2;
    view0.format = PixelFormat::GRAY8;

    ImageView view1;
    view1.data = pixels1.data();
    view1.width = 2;
    view1.height = 2;
    view1.stride = 2;
    view1.format = PixelFormat::GRAY8;

    ImageView view2;
    view2.data = pixels2.data();
    view2.width = 2;
    view2.height = 2;
    view2.stride = 2;
    view2.format = PixelFormat::GRAY8;

    Frame frame0;
    frame0.view = view0;
    frame0.frame_id = 0;
    frame0.timestamp_ns = 0;

    Frame frame1;
    frame1.view = view1;
    frame1.frame_id = 1;
    frame1.timestamp_ns = 100'000'000;

    Frame frame2;
    frame2.view = view2;
    frame2.frame_id = 2;
    frame2.timestamp_ns = 250'000'000;

    FakeFrameSource fake;

    fake.addFrame(frame0);
    fake.addFrame(frame1);
    fake.addFrame(frame2);

    /*
     * Important:
     *
     * Consumer uses FrameSource&, not FakeFrameSource.
     */
    FrameSource& source = fake;

    auto result0 = source.nextFrame();

    CHECK(result0.has_value());
    CHECK(result0->frame_id == 0);
    CHECK(result0->timestamp_ns == 0);
    CHECK(result0->view.data[0] == 10);

    auto result1 = source.nextFrame();

    CHECK(result1.has_value());
    CHECK(result1->frame_id == 1);
    CHECK(
        result1->timestamp_ns ==
        100'000'000
    );
    CHECK(result1->view.data[0] == 20);

    auto result2 = source.nextFrame();

    CHECK(result2.has_value());
    CHECK(result2->frame_id == 2);
    CHECK(
        result2->timestamp_ns ==
        250'000'000
    );
    CHECK(result2->view.data[0] == 30);

    /*
     * Normal end-of-stream.
     */
    auto end = source.nextFrame();

    CHECK(!end.has_value());

    return 0;
}