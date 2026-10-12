#include <edge_ai/preprocess/preprocess.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace {

size_t bytesPerPixel(PixelFormat format){
    switch(format){
        case PixelFormat::GRAY8: return 1;
        case PixelFormat::RGB24: return 3;
        case PixelFormat::BGR24: return 3;
        case PixelFormat::YUYV:  return 2;
    }
    return 0;
}

uint8_t clampByte(float v){
    if(v <= 0.0f) return 0;
    if(v >= 255.0f) return 255;
    return static_cast<uint8_t>(v + 0.5f);
}

// Reads one source pixel as RGB, respecting the view's stride and format.
void readRGB(const ImageView& view, size_t x, size_t y, float rgb[3]){
    const uint8_t* row = view.data + y*view.stride;

    switch(view.format){
        case PixelFormat::RGB24:{
            const uint8_t* p = row + 3*x;
            rgb[0] = p[0]; rgb[1] = p[1]; rgb[2] = p[2];
            return;
        }
        case PixelFormat::BGR24:{
            const uint8_t* p = row + 3*x;
            rgb[0] = p[2]; rgb[1] = p[1]; rgb[2] = p[0];
            return;
        }
        case PixelFormat::YUYV:{
            // Each 4-byte macropixel holds Y0 U Y1 V for two horizontal pixels.
            const uint8_t* macro = row + (x & ~static_cast<size_t>(1))*2;
            const float yv = (x & 1) ? macro[2] : macro[0];
            const float u = static_cast<float>(macro[1]) - 128.0f;
            const float v = static_cast<float>(macro[3]) - 128.0f;
            // BT.601 limited-range conversion, rounded like a real decoder output.
            const float c = 1.164f*(yv - 16.0f);
            rgb[0] = clampByte(c + 1.596f*v);
            rgb[1] = clampByte(c - 0.392f*u - 0.813f*v);
            rgb[2] = clampByte(c + 2.017f*u);
            return;
        }
        case PixelFormat::GRAY8:
            break;
    }
    throw std::invalid_argument("view format not supported");
}

struct AxisSample{
    size_t i0;
    size_t i1;
    float w1;
};

// OpenCV INTER_LINEAR convention: half-pixel centers, edge clamping.
std::vector<AxisSample> axisSamples(size_t src, size_t dst){
    std::vector<AxisSample> samples(dst);
    const float ratio = static_cast<float>(src) / static_cast<float>(dst);

    for(size_t i=0; i<dst; i++){
        float s = (static_cast<float>(i) + 0.5f)*ratio - 0.5f;
        if(s < 0.0f) s = 0.0f;

        size_t i0 = static_cast<size_t>(s);
        if(i0 >= src) i0 = src - 1;
        const size_t i1 = std::min(i0 + 1, src - 1);

        samples[i] = {i0, i1, s - static_cast<float>(i0)};
    }
    return samples;
}

} // namespace

PreprocessMetadata calculateLetterboxGeometry(size_t width, size_t height, size_t target_width, size_t target_height){
    if (width == 0) throw std::invalid_argument("zero source width");
    if (height == 0) throw std::invalid_argument("zero source height");
    if (target_width == 0) throw std::invalid_argument("zero target width");
    if (target_height == 0) throw std::invalid_argument("zero target height");

    constexpr size_t max_dim = std::numeric_limits<uint16_t>::max();
    if (width > max_dim || height > max_dim || target_width > max_dim || target_height > max_dim){
        throw std::invalid_argument("dimension too large");
    }

    PreprocessMetadata res;

    res.original_height = static_cast<uint16_t>(height);
    res.original_width = static_cast<uint16_t>(width);

    res.target_height = static_cast<uint16_t>(target_height);
    res.target_width = static_cast<uint16_t>(target_width);

    float scale_width = static_cast<float>(target_width) / width;
    float scale_height = static_cast<float>(target_height) / height;
    res.scale = std::min(scale_width, scale_height);

    // round() matches the Python reference letterbox.
    const long resized_width = std::max(1L, std::lround(width * res.scale));
    const long resized_height = std::max(1L, std::lround(height * res.scale));

    res.resized_width = static_cast<uint16_t>(std::min<long>(resized_width, target_width));
    res.resized_height = static_cast<uint16_t>(std::min<long>(resized_height, target_height));

    res.pad_left = (target_width - res.resized_width)/2;
    res.pad_top = (target_height - res.resized_height)/2;

    return res;
}

void preprocessInto(const ImageView& view, const PreprocessMetadata& meta, float* dst_chw){
    const size_t bpp = bytesPerPixel(view.format);

    if(view.format == PixelFormat::GRAY8 || bpp == 0) throw std::invalid_argument("view format not supported");
    if(view.data == nullptr) throw std::invalid_argument("null image data");
    if(view.width == 0 || view.height == 0) throw std::invalid_argument("zero image dimension");
    if(view.stride < view.width*bpp) throw std::invalid_argument("stride smaller than row size");
    if(view.format == PixelFormat::YUYV && view.width % 2 != 0) throw std::invalid_argument("YUYV width must be even");
    if(view.width != meta.original_width || view.height != meta.original_height){
        throw std::invalid_argument("metadata does not match image dimensions");
    }
    if(dst_chw == nullptr) throw std::invalid_argument("null destination tensor");

    const size_t tw = meta.target_width;
    const size_t th = meta.target_height;
    const size_t plane = tw*th;

    float* r_plane = dst_chw;
    float* g_plane = dst_chw + plane;
    float* b_plane = dst_chw + 2*plane;

    std::fill(dst_chw, dst_chw + 3*plane, static_cast<float>(kLetterboxPad) / 255.0f);

    const std::vector<AxisSample> xs = axisSamples(view.width, meta.resized_width);
    const std::vector<AxisSample> ys = axisSamples(view.height, meta.resized_height);

    for(size_t y=0; y<meta.resized_height; y++){
        const AxisSample& sy = ys[y];
        const size_t dst_row = (y + meta.pad_top)*tw + meta.pad_left;

        for(size_t x=0; x<meta.resized_width; x++){
            const AxisSample& sx = xs[x];

            float p00[3], p01[3], p10[3], p11[3];
            readRGB(view, sx.i0, sy.i0, p00);
            readRGB(view, sx.i1, sy.i0, p01);
            readRGB(view, sx.i0, sy.i1, p10);
            readRGB(view, sx.i1, sy.i1, p11);

            float out[3];
            for(int c=0; c<3; c++){
                const float top = p00[c] + (p01[c] - p00[c])*sx.w1;
                const float bottom = p10[c] + (p11[c] - p10[c])*sx.w1;
                out[c] = top + (bottom - top)*sy.w1;
            }

            const size_t idx = dst_row + x;
            r_plane[idx] = out[0] / 255.0f;
            g_plane[idx] = out[1] / 255.0f;
            b_plane[idx] = out[2] / 255.0f;
        }
    }
}

PreprocessResult preprocess(const ImageView& view, size_t target_width, size_t target_height){
    PreprocessResult result;
    result.meta = calculateLetterboxGeometry(view.width, view.height, target_width, target_height);
    result.tensor.resize(3*target_width*target_height);
    preprocessInto(view, result.meta, result.tensor.data());
    return result;
}

void modelToOriginal(float& x, float& y, const PreprocessMetadata& meta){
    x = (x - meta.pad_left) / meta.scale;
    y = (y - meta.pad_top) / meta.scale;

    x = std::clamp(x, 0.0f, static_cast<float>(meta.original_width));
    y = std::clamp(y, 0.0f, static_cast<float>(meta.original_height));
}
