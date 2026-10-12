#include <edge_ai/motion/motion_gate.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

#include <edge_ai/core/time.hpp>

namespace {

inline uint8_t grayAt(const ImageView& view, const uint8_t* row, size_t x){
    switch(view.format){
        case PixelFormat::GRAY8: return row[x];
        case PixelFormat::YUYV:  return row[2*x];   // luma is already grayscale
        case PixelFormat::RGB24:{
            const uint8_t* p = row + 3*x;
            return static_cast<uint8_t>((77u*p[0] + 150u*p[1] + 29u*p[2]) >> 8);
        }
        case PixelFormat::BGR24:{
            const uint8_t* p = row + 3*x;
            return static_cast<uint8_t>((77u*p[2] + 150u*p[1] + 29u*p[0]) >> 8);
        }
    }
    return 0;
}

size_t bytesPerPixel(PixelFormat format){
    switch(format){
        case PixelFormat::GRAY8: return 1;
        case PixelFormat::YUYV:  return 2;
        case PixelFormat::RGB24: return 3;
        case PixelFormat::BGR24: return 3;
    }
    return 0;
}

} // namespace

MotionGate::MotionGate(const MotionConfig& config)
    : config_(config)
{
    if(config_.width == 0 || config_.height == 0) throw std::invalid_argument("zero motion resolution");

    const size_t n = config_.width*config_.height;
    current_.assign(n, 0);
    previous_.assign(n, 0);
    mask_.assign(n, 0);
}

void MotionGate::reset(){
    has_previous_ = false;
    frames_seen_ = 0;
    last_event_ns_.reset();
    last_timestamp_ns_.reset();
    last_changed_pixels_ = 0;
    std::fill(mask_.begin(), mask_.end(), 0);
}

void MotionGate::downsampleGray(const ImageView& view){
    const size_t ow = config_.width;
    const size_t oh = config_.height;

    // Area average: each output cell averages its source rectangle.
    for(size_t oy=0; oy<oh; oy++){
        const size_t y0 = oy*view.height/oh;
        const size_t y1 = std::max(y0 + 1, (oy + 1)*view.height/oh);

        for(size_t ox=0; ox<ow; ox++){
            const size_t x0 = ox*view.width/ow;
            const size_t x1 = std::max(x0 + 1, (ox + 1)*view.width/ow);

            uint32_t sum = 0;
            for(size_t y=y0; y<y1; y++){
                const uint8_t* row = view.data + y*view.stride;
                for(size_t x=x0; x<x1; x++) sum += grayAt(view, row, x);
            }
            const uint32_t count = static_cast<uint32_t>((y1 - y0)*(x1 - x0));
            current_[oy*ow + ox] = static_cast<uint8_t>((sum + count/2) / count);
        }
    }
}

MotionResult MotionGate::update(const ImageView& view, uint64_t timestamp_ns){
    const uint64_t started = now_monotonic_ns();

    if(view.data == nullptr || view.width == 0 || view.height == 0) throw std::invalid_argument("invalid motion input");
    if(view.stride < view.width*bytesPerPixel(view.format)) throw std::invalid_argument("stride smaller than row size");

    // A restarted/looped stream moves time backwards: treat it as a new stream.
    if(last_timestamp_ns_.has_value() && timestamp_ns < *last_timestamp_ns_) reset();
    last_timestamp_ns_ = timestamp_ns;

    source_width_ = view.width;
    source_height_ = view.height;

    downsampleGray(view);

    MotionResult result;
    ++frames_seen_;

    // Frame 1 only establishes the reference image.
    if(!has_previous_){
        previous_.swap(current_);
        has_previous_ = true;
        std::fill(mask_.begin(), mask_.end(), 0);
        last_changed_pixels_ = 0;
        result.processing_ms = static_cast<float>(now_monotonic_ns() - started) / 1e6f;
        return result;
    }

    size_t changed = 0;
    for(size_t i=0; i<current_.size(); i++){
        const int delta = std::abs(static_cast<int>(current_[i]) - static_cast<int>(previous_[i]));
        const uint8_t on = delta >= config_.pixel_threshold ? 1 : 0;
        mask_[i] = on;
        changed += on;
    }

    last_changed_pixels_ = changed;
    result.changed_pixels = changed;
    result.score = static_cast<float>(changed) / static_cast<float>(current_.size());

    // Always compare adjacent frames: we measure change, not distance from a keyframe.
    previous_.swap(current_);

    if(frames_seen_ <= config_.warmup_frames){
        result.processing_ms = static_cast<float>(now_monotonic_ns() - started) / 1e6f;
        return result;
    }

    result.active = changed >= config_.minimum_changed_area;

    const bool cooldown_elapsed =
        !last_event_ns_.has_value() ||
        timestamp_ns - *last_event_ns_ >= static_cast<uint64_t>(config_.cooldown_ms)*1'000'000ULL;

    result.event = result.active && cooldown_elapsed;
    if(result.event) last_event_ns_ = timestamp_ns;

    result.processing_ms = static_cast<float>(now_monotonic_ns() - started) / 1e6f;
    return result;
}

float MotionGate::changedFractionOutside(const std::vector<Detection>& boxes) const{
    if(last_changed_pixels_ == 0 || source_width_ == 0 || source_height_ == 0) return 0.0f;

    const float sx = static_cast<float>(config_.width) / static_cast<float>(source_width_);
    const float sy = static_cast<float>(config_.height) / static_cast<float>(source_height_);

    struct GridBox{ float x1, y1, x2, y2; };
    std::vector<GridBox> grid;
    grid.reserve(boxes.size());
    for(const Detection& b : boxes){
        const float mx = (b.x2 - b.x1)*config_.box_margin;
        const float my = (b.y2 - b.y1)*config_.box_margin;
        grid.push_back({(b.x1 - mx)*sx, (b.y1 - my)*sy, (b.x2 + mx)*sx, (b.y2 + my)*sy});
    }

    size_t outside = 0;
    for(size_t y=0; y<config_.height; y++){
        for(size_t x=0; x<config_.width; x++){
            if(!mask_[y*config_.width + x]) continue;

            const float cx = static_cast<float>(x) + 0.5f;
            const float cy = static_cast<float>(y) + 0.5f;
            bool inside = false;
            for(const GridBox& g : grid){
                if(cx >= g.x1 && cx <= g.x2 && cy >= g.y1 && cy <= g.y2){ inside = true; break; }
            }
            if(!inside) ++outside;
        }
    }
    return static_cast<float>(outside) / static_cast<float>(last_changed_pixels_);
}

size_t MotionGate::changedPixelsInside(const Detection& box) const{
    if(last_changed_pixels_ == 0 || source_width_ == 0 || source_height_ == 0) return 0;

    const float sx = static_cast<float>(config_.width) / static_cast<float>(source_width_);
    const float sy = static_cast<float>(config_.height) / static_cast<float>(source_height_);

    const long x0 = std::max(0L, static_cast<long>(std::floor(box.x1*sx)));
    const long y0 = std::max(0L, static_cast<long>(std::floor(box.y1*sy)));
    const long x1 = std::min(static_cast<long>(config_.width), static_cast<long>(std::ceil(box.x2*sx)));
    const long y1 = std::min(static_cast<long>(config_.height), static_cast<long>(std::ceil(box.y2*sy)));

    size_t count = 0;
    for(long y=y0; y<y1; y++){
        for(long x=x0; x<x1; x++) count += mask_[static_cast<size_t>(y)*config_.width + static_cast<size_t>(x)];
    }
    return count;
}
