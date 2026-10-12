#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <edge_ai/preprocess/preprocess.hpp>

#include "test_common.hpp"

namespace {

constexpr float kEps = 1e-5f;
constexpr float kPad = kLetterboxPad / 255.0f;

template <typename Fn>
bool throwsInvalid(Fn fn){
    try{
        fn();
    }
    catch(const std::invalid_argument&){
        return true;
    }
    return false;
}

float at(const PreprocessResult& r, size_t channel, size_t x, size_t y){
    const size_t w = r.meta.target_width;
    const size_t h = r.meta.target_height;
    return r.tensor[channel*w*h + y*w + x];
}

int geometryTests(){
    // 1280x720 -> 640x640: landscape, vertical padding.
    {
        const PreprocessMetadata m = calculateLetterboxGeometry(1280, 720, 640, 640);
        CHECK_NEAR(m.scale, 0.5f, kEps);
        CHECK(m.original_width == 1280 && m.original_height == 720);
        CHECK(m.target_width == 640 && m.target_height == 640);
        CHECK(m.resized_width == 640 && m.resized_height == 360);
        CHECK(m.pad_left == 0 && m.pad_top == 140);
    }
    // 640x480 -> 320x320
    {
        const PreprocessMetadata m = calculateLetterboxGeometry(640, 480, 320, 320);
        CHECK_NEAR(m.scale, 0.5f, kEps);
        CHECK(m.resized_width == 320 && m.resized_height == 240);
        CHECK(m.pad_left == 0 && m.pad_top == 40);
    }
    // 720x1280 -> 640x640: portrait, horizontal padding.
    {
        const PreprocessMetadata m = calculateLetterboxGeometry(720, 1280, 640, 640);
        CHECK_NEAR(m.scale, 0.5f, kEps);
        CHECK(m.resized_width == 360 && m.resized_height == 640);
        CHECK(m.pad_left == 140 && m.pad_top == 0);
    }
    // 640x640 -> 640x640: identity.
    {
        const PreprocessMetadata m = calculateLetterboxGeometry(640, 640, 640, 640);
        CHECK_NEAR(m.scale, 1.0f, kEps);
        CHECK(m.resized_width == 640 && m.resized_height == 640);
        CHECK(m.pad_left == 0 && m.pad_top == 0);
    }
    // 800x600 -> 320x320
    {
        const PreprocessMetadata m = calculateLetterboxGeometry(800, 600, 320, 320);
        CHECK_NEAR(m.scale, 0.4f, kEps);
        CHECK(m.resized_width == 320 && m.resized_height == 240);
        CHECK(m.pad_left == 0 && m.pad_top == 40);
    }
    // Replay test clip into the deployed 320x256 (W x H) model: rounding matches Python.
    {
        const PreprocessMetadata m = calculateLetterboxGeometry(1620, 1080, 320, 256);
        CHECK_NEAR(m.scale, 320.0f/1620.0f, kEps);
        CHECK(m.resized_width == 320);
        CHECK(m.resized_height == 213);   // round(213.33)
        CHECK(m.pad_left == 0 && m.pad_top == 21);
    }
    // 640x480 camera into 320x256.
    {
        const PreprocessMetadata m = calculateLetterboxGeometry(640, 480, 320, 256);
        CHECK_NEAR(m.scale, 0.5f, kEps);
        CHECK(m.resized_width == 320 && m.resized_height == 240);
        CHECK(m.pad_left == 0 && m.pad_top == 8);
    }
    return 0;
}

int invalidInputTests(){
    CHECK(throwsInvalid([]{ calculateLetterboxGeometry(0, 480, 320, 320); }));
    CHECK(throwsInvalid([]{ calculateLetterboxGeometry(640, 0, 320, 320); }));
    CHECK(throwsInvalid([]{ calculateLetterboxGeometry(640, 480, 0, 320); }));
    CHECK(throwsInvalid([]{ calculateLetterboxGeometry(640, 480, 320, 0); }));

    TestImage gray = makeImage(4, 4, PixelFormat::GRAY8, 0, [](size_t, size_t, uint8_t* p){ *p = 1; });
    CHECK(throwsInvalid([&]{ preprocess(gray.view, 4, 4); }));

    TestImage rgb = makeSolidRGB(4, 4, 1, 2, 3);
    ImageView bad_stride = rgb.view;
    bad_stride.stride = 11;
    CHECK(throwsInvalid([&]{ preprocess(bad_stride, 4, 4); }));

    ImageView null_data = rgb.view;
    null_data.data = nullptr;
    CHECK(throwsInvalid([&]{ preprocess(null_data, 4, 4); }));

    TestImage odd_yuyv = makeImage(3, 2, PixelFormat::YUYV, 2, [](size_t, size_t, uint8_t* p){ p[0] = 16; p[1] = 128; });
    CHECK(throwsInvalid([&]{ preprocess(odd_yuyv.view, 4, 4); }));

    // Metadata computed for a different frame size must be rejected.
    PreprocessMetadata other = calculateLetterboxGeometry(8, 8, 4, 4);
    std::vector<float> dst(3*4*4);
    CHECK(throwsInvalid([&]{ preprocessInto(rgb.view, other, dst.data()); }));
    return 0;
}

int strideTest(){
    // 2x2 RGB with 5 bytes of 0xEE padding per row; identity resize must read only pixels.
    TestImage img = makeImage(2, 2, PixelFormat::RGB24, 5, [](size_t x, size_t y, uint8_t* p){
        const uint8_t base = static_cast<uint8_t>(10*(y*2 + x));
        p[0] = base + 1; p[1] = base + 2; p[2] = base + 3;
    });
    CHECK(img.view.stride == 11);

    const PreprocessResult r = preprocess(img.view, 2, 2);
    for(size_t y=0; y<2; y++){
        for(size_t x=0; x<2; x++){
            const float base = static_cast<float>(10*(y*2 + x));
            CHECK_NEAR(at(r, 0, x, y), (base + 1)/255.0f, kEps);
            CHECK_NEAR(at(r, 1, x, y), (base + 2)/255.0f, kEps);
            CHECK_NEAR(at(r, 2, x, y), (base + 3)/255.0f, kEps);
        }
    }
    return 0;
}

int hwcToChwTest(){
    // A=(1,2,3) B=(4,5,6) C=(7,8,9) D=(10,11,12)
    TestImage img = makeImage(2, 2, PixelFormat::RGB24, 0, [](size_t x, size_t y, uint8_t* p){
        const uint8_t base = static_cast<uint8_t>(3*(y*2 + x));
        p[0] = base + 1; p[1] = base + 2; p[2] = base + 3;
    });

    const PreprocessResult r = preprocess(img.view, 2, 2);
    CHECK(r.tensor.size() == 12);

    const float expected[12] = {1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12};
    for(size_t i=0; i<12; i++) CHECK_NEAR(r.tensor[i], expected[i]/255.0f, kEps);
    return 0;
}

int normalizationTest(){
    TestImage img = makeImage(3, 1, PixelFormat::RGB24, 0, [](size_t x, size_t, uint8_t* p){
        const uint8_t v = x == 0 ? 0 : (x == 1 ? 51 : 255);
        p[0] = v; p[1] = v; p[2] = v;
    });

    const PreprocessResult r = preprocess(img.view, 3, 1);
    CHECK_NEAR(at(r, 0, 0, 0), 0.0f, kEps);
    CHECK_NEAR(at(r, 1, 1, 0), 0.2f, kEps);
    CHECK_NEAR(at(r, 2, 2, 0), 1.0f, kEps);
    return 0;
}

int bilinearResizeTest(){
    // R channel: [0 100; 200 40], upscaled 2x2 -> 4x4 (OpenCV half-pixel convention).
    TestImage img = makeImage(2, 2, PixelFormat::RGB24, 0, [](size_t x, size_t y, uint8_t* p){
        const uint8_t values[2][2] = {{0, 100}, {200, 40}};
        p[0] = values[y][x]; p[1] = 0; p[2] = 0;
    });

    const PreprocessResult r = preprocess(img.view, 4, 4);
    CHECK(r.meta.resized_width == 4 && r.meta.resized_height == 4);

    // Row 0 interpolates 0 -> 100 with weights 0, .25, .75, 1.
    CHECK_NEAR(at(r, 0, 0, 0)*255.0f, 0.0f, 1e-3);
    CHECK_NEAR(at(r, 0, 1, 0)*255.0f, 25.0f, 1e-3);
    CHECK_NEAR(at(r, 0, 2, 0)*255.0f, 75.0f, 1e-3);
    CHECK_NEAR(at(r, 0, 3, 0)*255.0f, 100.0f, 1e-3);

    // Column 0 interpolates 0 -> 200.
    CHECK_NEAR(at(r, 0, 0, 1)*255.0f, 50.0f, 1e-3);
    CHECK_NEAR(at(r, 0, 0, 2)*255.0f, 150.0f, 1e-3);

    // Interior point (x=1, y=1): weights .25/.25 -> 0*.5625 + 100*.1875 + 200*.1875 + 40*.0625
    CHECK_NEAR(at(r, 0, 1, 1)*255.0f, 58.75f, 1e-3);

    // Downscale 4x4 -> 2x2 of a uniform image keeps the value.
    TestImage solid = makeSolidRGB(4, 4, 90, 120, 150);
    const PreprocessResult d = preprocess(solid.view, 2, 2);
    CHECK_NEAR(at(d, 1, 1, 1)*255.0f, 120.0f, 1e-3);
    return 0;
}

int letterboxTest(){
    // 4x2 image into 4x4: scale 1, resized 4x2, pad_top 1, rows 0 and 3 are padding.
    TestImage img = makeSolidRGB(4, 2, 10, 20, 30);
    const PreprocessResult r = preprocess(img.view, 4, 4);

    CHECK(r.meta.pad_top == 1 && r.meta.pad_left == 0);
    for(size_t c=0; c<3; c++){
        for(size_t x=0; x<4; x++){
            CHECK_NEAR(at(r, c, x, 0), kPad, kEps);
            CHECK_NEAR(at(r, c, x, 3), kPad, kEps);
        }
    }
    for(size_t y=1; y<=2; y++){
        for(size_t x=0; x<4; x++){
            CHECK_NEAR(at(r, 0, x, y), 10/255.0f, kEps);
            CHECK_NEAR(at(r, 1, x, y), 20/255.0f, kEps);
            CHECK_NEAR(at(r, 2, x, y), 30/255.0f, kEps);
        }
    }

    // Portrait 2x4 into 4x4: horizontal padding columns 0 and 3.
    TestImage tall = makeSolidRGB(2, 4, 200, 200, 200);
    const PreprocessResult t = preprocess(tall.view, 4, 4);
    CHECK(t.meta.pad_left == 1 && t.meta.pad_top == 0);
    CHECK_NEAR(at(t, 0, 0, 2), kPad, kEps);
    CHECK_NEAR(at(t, 0, 3, 2), kPad, kEps);
    CHECK_NEAR(at(t, 0, 1, 2), 200/255.0f, kEps);
    return 0;
}

int formatParityTest(){
    // BGR24 input must produce the same RGB tensor as RGB24.
    auto fill = [](size_t x, size_t y, uint8_t* p, bool bgr){
        const uint8_t r = static_cast<uint8_t>(20 + 30*x);
        const uint8_t g = static_cast<uint8_t>(40 + 10*y);
        const uint8_t b = static_cast<uint8_t>(200 - 20*x);
        p[0] = bgr ? b : r; p[1] = g; p[2] = bgr ? r : b;
    };
    TestImage rgb = makeImage(4, 3, PixelFormat::RGB24, 2, [&](size_t x, size_t y, uint8_t* p){ fill(x, y, p, false); });
    TestImage bgr = makeImage(4, 3, PixelFormat::BGR24, 2, [&](size_t x, size_t y, uint8_t* p){ fill(x, y, p, true); });

    const PreprocessResult a = preprocess(rgb.view, 8, 8);
    const PreprocessResult b = preprocess(bgr.view, 8, 8);
    for(size_t i=0; i<a.tensor.size(); i++) CHECK_NEAR(a.tensor[i], b.tensor[i], kEps);

    // YUYV neutral chroma: Y=16 -> black, Y=235 -> white, Y=126 -> 128.
    TestImage yuyv = makeImage(4, 1, PixelFormat::YUYV, 4, [](size_t x, size_t, uint8_t* p){
        const uint8_t luma[4] = {16, 235, 126, 126};
        p[0] = luma[x];
        p[1] = 128;   // U for even pixels, V for odd pixels
    });
    const PreprocessResult y = preprocess(yuyv.view, 4, 1);
    for(size_t c=0; c<3; c++){
        CHECK_NEAR(at(y, c, 0, 0)*255.0f, 0.0f, 1e-3);
        CHECK_NEAR(at(y, c, 1, 0)*255.0f, 255.0f, 1e-3);
        CHECK_NEAR(at(y, c, 2, 0)*255.0f, 128.0f, 1e-3);
    }
    return 0;
}

int reverseMappingTest(){
    const PreprocessMetadata m = calculateLetterboxGeometry(1620, 1080, 320, 256);

    const float box[4] = {100.0f, 200.0f, 500.0f, 900.0f};
    float x1 = box[0]*m.scale + m.pad_left;
    float y1 = box[1]*m.scale + m.pad_top;
    float x2 = box[2]*m.scale + m.pad_left;
    float y2 = box[3]*m.scale + m.pad_top;

    modelToOriginal(x1, y1, m);
    modelToOriginal(x2, y2, m);
    CHECK_NEAR(x1, box[0], 1e-2);
    CHECK_NEAR(y1, box[1], 1e-2);
    CHECK_NEAR(x2, box[2], 1e-2);
    CHECK_NEAR(y2, box[3], 1e-2);

    // Points in the padding area clamp to the image.
    float px = -5.0f, py = 5.0f;
    modelToOriginal(px, py, m);
    CHECK(px == 0.0f && py == 0.0f);

    float qx = 400.0f, qy = 255.0f;
    modelToOriginal(qx, qy, m);
    CHECK(qx == 1620.0f && qy == 1080.0f);
    return 0;
}

} // namespace

int main(){
    if(geometryTests()) return 1;
    if(invalidInputTests()) return 1;
    if(strideTest()) return 1;
    if(hwcToChwTest()) return 1;
    if(normalizationTest()) return 1;
    if(bilinearResizeTest()) return 1;
    if(letterboxTest()) return 1;
    if(formatParityTest()) return 1;
    if(reverseMappingTest()) return 1;

    std::cout << "All preprocessing tests passed.\n";
    return 0;
}
